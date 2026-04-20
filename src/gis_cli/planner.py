from __future__ import annotations

import json
import re
from pathlib import Path
from urllib import request

from .catalog import IntentCatalog
from .models import PlanStep, TaskPlan


class IntentPlanner:
    def __init__(
        self,
        catalog: IntentCatalog | None = None,
        llm_rag_config_path: Path | None = None,
    ) -> None:
        self.catalog = catalog or IntentCatalog()
        self.llm_rag_config_path = llm_rag_config_path or Path("config") / "llm_rag_config.json"
        self.llm_rag_config = self._load_llm_rag_config()
        self._last_rag_source_count = 0
        self._last_rag_backend = "native"
        self._last_rag_backend_error = ""

    def build_plan(self, prompt: str) -> TaskPlan:
        rule_plan = self.build_plan_hybrid(prompt)
        if not self._llm_rag_enabled():
            return rule_plan

        llm_plan = self._build_plan_with_llm_rag(prompt, rule_plan)
        return llm_plan or rule_plan

    def build_plan_hybrid(self, prompt: str, min_rule_confidence: float = 0.22) -> TaskPlan:
        prompt_lc = prompt.strip().lower()
        scores: list[tuple[float, object]] = []

        for intent in self.catalog.all():
            hits = sum(1 for kw in intent.keywords if kw.lower() in prompt_lc)
            total = max(len(intent.keywords), 1)
            confidence = hits / total

            # Boost common primary verbs so mixed prompts map to the right first-stage intent.
            if intent.name == "data_integration" and any(
                k in prompt_lc for k in ("整合", "入库", "合并")
            ):
                confidence += 0.25
            if intent.name == "thematic_mapping" and any(
                k in prompt_lc for k in ("专题图", "制图", "版式")
            ):
                confidence += 0.15
            if intent.name == "spatial_analysis" and any(
                k in prompt_lc for k in ("缓冲", "叠加", "空间")
            ):
                confidence += 0.2

            scores.append((confidence, intent))

        scores.sort(key=lambda x: x[0], reverse=True)
        best_conf, best = scores[0]

        planner_mode = "rule"
        if best_conf < min_rule_confidence:
            best = self._fallback_pick_intent(prompt_lc)
            planner_mode = "fallback"
            best_conf = max(best_conf, 0.2)

        missing_parameters = self._guess_missing_parameters(prompt_lc, best.name)
        clarifying_questions = [
            f"请补充：{item}"
            for item in missing_parameters
        ]
        steps = [
            PlanStep(order=i + 1, title=title)
            for i, title in enumerate(best.steps)
        ]

        return TaskPlan(
            intent=best.name,
            confidence=round(best_conf, 3),
            planner_mode=planner_mode,
            missing_parameters=missing_parameters,
            clarifying_questions=clarifying_questions,
            steps=steps,
        )

    def _fallback_pick_intent(self, prompt: str):
        # LLM fallback placeholder: deterministic intent router for low-confidence prompts.
        if any(k in prompt for k in ("缓冲", "叠加", "相交", "空间")):
            target = "spatial_analysis"
        elif any(k in prompt for k in ("制图", "图例", "标注", "版式", "导出图")):
            target = "thematic_mapping"
        elif any(k in prompt for k in ("质量", "拓扑", "检查", "校验")):
            target = "quality_check"
        elif any(k in prompt for k in ("批量", "分幅", "循环")):
            target = "batch_processing"
        else:
            target = "data_integration"

        for intent in self.catalog.all():
            if intent.name == target:
                return intent
        return self.catalog.all()[0]

    def _guess_missing_parameters(self, prompt: str, intent_name: str) -> list[str]:
        missing: list[str] = []

        if "输出" not in prompt and "导出" not in prompt:
            missing.append("输出路径或输出文件名")

        if intent_name in {"data_integration", "spatial_analysis"}:
            if "投影" not in prompt and "坐标" not in prompt:
                missing.append("目标坐标系统")

        if intent_name == "data_integration":
            if "数据库" not in prompt and "gdb" not in prompt:
                missing.append("目标数据库名称")

        if intent_name == "spatial_analysis":
            if "缓冲" in prompt and ("米" not in prompt and "公里" not in prompt):
                missing.append("缓冲距离与单位")

        return missing

    def _llm_rag_enabled(self) -> bool:
        return bool(self.llm_rag_config.get("enabled", False))

    def _load_llm_rag_config(self) -> dict:
        if not self.llm_rag_config_path.exists():
            return {}
        try:
            payload = json.loads(self.llm_rag_config_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def reload_llm_rag_config(self) -> dict:
        self.llm_rag_config = self._load_llm_rag_config()
        return self.get_llm_rag_status()

    def get_llm_rag_status(self) -> dict:
        llm_cfg = self.llm_rag_config.get("llm", {})
        rag_cfg = self.llm_rag_config.get("rag", {})
        static_paths = rag_cfg.get("knowledge_paths", []) if isinstance(rag_cfg, dict) else []
        static_paths = static_paths if isinstance(static_paths, list) else []
        dynamic_paths = self._discover_dynamic_knowledge_paths(rag_cfg if isinstance(rag_cfg, dict) else {})

        return {
            "enabled": self._llm_rag_enabled(),
            "config_path": str(self.llm_rag_config_path),
            "model": str(llm_cfg.get("model", "")) if isinstance(llm_cfg, dict) else "",
            "api_base": str(llm_cfg.get("api_base", "")) if isinstance(llm_cfg, dict) else "",
            "has_api_key": bool(str(llm_cfg.get("api_key", "")).strip()) if isinstance(llm_cfg, dict) else False,
            "static_source_count": len(static_paths),
            "dynamic_source_count": len(dynamic_paths),
            "rag_top_k": int(rag_cfg.get("top_k", 3)) if isinstance(rag_cfg, dict) else 3,
            "rag_chunk_size": int(rag_cfg.get("chunk_size", 1200)) if isinstance(rag_cfg, dict) else 1200,
            "rag_backend": str(rag_cfg.get("backend", "native")) if isinstance(rag_cfg, dict) else "native",
            "rag_backend_active": self._last_rag_backend,
            "rag_backend_error": self._last_rag_backend_error,
            "static_sources": [str(x) for x in static_paths],
            "dynamic_sources_preview": [str(x) for x in dynamic_paths[:10]],
        }

    def _build_plan_with_llm_rag(self, prompt: str, rule_plan: TaskPlan) -> TaskPlan | None:
        llm_cfg = self.llm_rag_config.get("llm", {})
        if not isinstance(llm_cfg, dict):
            return None

        api_key = str(llm_cfg.get("api_key", "")).strip()
        model = str(llm_cfg.get("model", "")).strip()
        api_base = str(llm_cfg.get("api_base", "")).strip()
        if not api_key or not model or not api_base:
            return None

        rag_context = self._retrieve_rag_context(prompt)
        intent_names = [i.name for i in self.catalog.all()]
        intent_steps = {i.name: i.steps for i in self.catalog.all()}

        planner_prompt = (
            "你是GIS任务规划器。请根据用户需求输出JSON，不要输出其他文本。\n"
            f"可选intent: {intent_names}\n"
            f"规则基线意图: {rule_plan.intent}, 规则置信度: {rule_plan.confidence}\n"
            "RAG上下文如下:\n"
            f"{rag_context}\n\n"
            "返回JSON结构:\n"
            "{\n"
            '  "intent": "data_integration|thematic_mapping|spatial_analysis|batch_processing|quality_check",\n'
            '  "confidence": 0.0,\n'
            '  "missing_parameters": ["..."],\n'
            '  "clarifying_questions": ["..."],\n'
            '  "steps": ["步骤1", "步骤2"]\n'
            "}\n"
            f"用户需求: {prompt}\n"
        )

        content = self._request_openai_compatible(api_base, api_key, model, planner_prompt)
        if not content:
            return None

        parsed = self._parse_llm_json(content)
        if not isinstance(parsed, dict):
            return None

        intent = str(parsed.get("intent", "")).strip()
        if intent not in intent_steps:
            return None

        try:
            confidence = float(parsed.get("confidence", rule_plan.confidence))
        except Exception:
            confidence = rule_plan.confidence
        confidence = max(0.0, min(1.0, confidence))

        missing_parameters_raw = parsed.get("missing_parameters", [])
        clarifying_questions_raw = parsed.get("clarifying_questions", [])
        steps_raw = parsed.get("steps", [])

        missing_parameters = [str(x) for x in missing_parameters_raw] if isinstance(missing_parameters_raw, list) else []
        clarifying_questions = [str(x) for x in clarifying_questions_raw] if isinstance(clarifying_questions_raw, list) else []
        step_titles = [str(x) for x in steps_raw] if isinstance(steps_raw, list) else []
        if not step_titles:
            step_titles = list(intent_steps[intent])

        steps = [PlanStep(order=i + 1, title=title) for i, title in enumerate(step_titles)]
        return TaskPlan(
            intent=intent,
            confidence=round(confidence, 3),
            planner_mode="rag_llm",
            missing_parameters=missing_parameters,
            clarifying_questions=clarifying_questions,
            steps=steps,
        )

    def _retrieve_rag_context(self, prompt: str) -> str:
        rag_cfg = self.llm_rag_config.get("rag", {})
        if not isinstance(rag_cfg, dict):
            rag_cfg = {}
        backend = str(rag_cfg.get("backend", "native")).strip().lower()
        self._last_rag_backend_error = ""

        if backend in {"langchain", "lc"}:
            context = self._retrieve_rag_context_langchain(prompt, rag_cfg)
            if context:
                self._last_rag_backend = "langchain"
                return context
            self._last_rag_backend = "native_fallback"
            return self._retrieve_rag_context_native(prompt, rag_cfg)

        self._last_rag_backend = "native"
        return self._retrieve_rag_context_native(prompt, rag_cfg)

    def _retrieve_rag_context_native(self, prompt: str, rag_cfg: dict) -> str:
        merged_paths = self._resolve_rag_paths(rag_cfg)
        top_k = int(rag_cfg.get("top_k", 3))
        chunk_size = int(rag_cfg.get("chunk_size", 1200))

        chunks: list[str] = []
        for p in merged_paths:
            try:
                path = Path(str(p))
                if not path.exists() or not path.is_file():
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
                for i in range(0, len(text), chunk_size):
                    segment = text[i : i + chunk_size].strip()
                    if segment:
                        chunks.append(segment)
            except Exception:
                continue

        self._last_rag_source_count = len(merged_paths)

        if not chunks:
            return ""

        query_tokens = self._tokenize(prompt)
        scored: list[tuple[int, str]] = []
        for c in chunks:
            score = self._score_chunk(query_tokens, c)
            if score > 0:
                scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [c for _, c in scored[: max(1, top_k)]]
        return "\n\n---\n\n".join(selected)

    def _retrieve_rag_context_langchain(self, prompt: str, rag_cfg: dict) -> str:
        merged_paths = self._resolve_rag_paths(rag_cfg)
        self._last_rag_source_count = len(merged_paths)
        if not merged_paths:
            return ""

        try:
            from langchain_community.document_loaders import TextLoader
            from langchain_community.vectorstores import Chroma
            from langchain_openai import OpenAIEmbeddings
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except Exception as exc:
            self._last_rag_backend_error = f"langchain_import_failed: {exc}"
            return ""

        docs = []
        use_unstructured = bool(rag_cfg.get("use_unstructured_loader", False))
        unstructured_loader_cls = None
        if use_unstructured:
            try:
                from langchain_community.document_loaders import UnstructuredFileLoader

                unstructured_loader_cls = UnstructuredFileLoader
            except Exception:
                unstructured_loader_cls = None

        for item in merged_paths:
            try:
                p = Path(str(item))
                if not p.exists() or not p.is_file():
                    continue
                if unstructured_loader_cls is not None:
                    loaded = unstructured_loader_cls(str(p)).load()
                else:
                    loaded = TextLoader(str(p), encoding="utf-8", autodetect_encoding=True).load()
                docs.extend(loaded)
            except Exception:
                continue

        if not docs:
            return ""

        chunk_size = int(rag_cfg.get("chunk_size", 1200))
        chunk_overlap = int(rag_cfg.get("chunk_overlap", 120))
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_docs = splitter.split_documents(docs)
        if not split_docs:
            return ""

        llm_cfg = self.llm_rag_config.get("llm", {})
        emb_cfg = rag_cfg.get("embeddings", {})
        if not isinstance(llm_cfg, dict):
            llm_cfg = {}
        if not isinstance(emb_cfg, dict):
            emb_cfg = {}

        api_key = str(emb_cfg.get("api_key") or llm_cfg.get("api_key", "")).strip()
        base_url = str(emb_cfg.get("api_base") or llm_cfg.get("api_base", "")).strip()
        emb_model = str(emb_cfg.get("model", "text-embedding-3-small")).strip()
        if not api_key or not base_url:
            self._last_rag_backend_error = "langchain_embeddings_missing_api_config"
            return ""

        try:
            embeddings = OpenAIEmbeddings(
                model=emb_model,
                api_key=api_key,
                base_url=base_url.rstrip("/"),
            )
            vectorstore = Chroma.from_documents(
                split_docs,
                embedding=embeddings,
                collection_name="gis_cli_rag_tmp",
            )
            top_k = int(rag_cfg.get("top_k", 3))
            if bool(rag_cfg.get("use_contextual_compression", False)):
                try:
                    from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
                    from langchain.retrievers.document_compressors import EmbeddingsFilter

                    base_retriever = vectorstore.as_retriever(search_kwargs={"k": max(2, top_k * 2)})
                    compressor = EmbeddingsFilter(
                        embeddings=embeddings,
                        similarity_threshold=float(rag_cfg.get("compression_similarity_threshold", 0.3)),
                    )
                    retriever = ContextualCompressionRetriever(
                        base_compressor=compressor,
                        base_retriever=base_retriever,
                    )
                    result_docs = retriever.invoke(prompt)
                except Exception:
                    result_docs = vectorstore.similarity_search(prompt, k=max(1, top_k))
            else:
                result_docs = vectorstore.similarity_search(prompt, k=max(1, top_k))

            text_chunks = [str(getattr(d, "page_content", "")).strip() for d in result_docs]
            text_chunks = [t for t in text_chunks if t]
            return "\n\n---\n\n".join(text_chunks)
        except Exception as exc:
            self._last_rag_backend_error = f"langchain_retrieval_failed: {exc}"
            return ""
        finally:
            try:
                vectorstore.delete_collection()  # type: ignore[name-defined]
            except Exception:
                pass

    def _resolve_rag_paths(self, rag_cfg: dict) -> list[str]:
        paths = rag_cfg.get(
            "knowledge_paths",
            [
                "README.md",
                "config/intents.json",
                "config/recovery_strategies.json",
            ],
        )
        if not isinstance(paths, list):
            paths = []

        dynamic_paths = self._discover_dynamic_knowledge_paths(rag_cfg)
        merged_paths: list[str] = []
        seen: set[str] = set()
        for item in [*paths, *dynamic_paths]:
            k = str(item)
            if k not in seen:
                seen.add(k)
                merged_paths.append(k)
        return merged_paths

    def _discover_dynamic_knowledge_paths(self, rag_cfg: dict) -> list[str]:
        if not bool(rag_cfg.get("auto_include_outputs", True)):
            return []

        output_root = Path(str(rag_cfg.get("output_root", "outputs")))
        if not output_root.exists() or not output_root.is_dir():
            return []

        max_files = int(rag_cfg.get("max_output_files", 30))
        allow_globs = rag_cfg.get(
            "output_globs",
            [
                "**/checklist.json",
                "**/arcpy_report.txt",
                "**/quality_report.json",
                "**/execution_report.txt",
                "**/export_summary.txt",
            ],
        )
        if not isinstance(allow_globs, list):
            allow_globs = []

        found: list[Path] = []
        for pattern in allow_globs:
            try:
                found.extend(output_root.glob(str(pattern)))
            except Exception:
                continue

        found = [p for p in found if p.is_file()]
        found.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
        trimmed = found[: max(1, max_files)]
        return [str(p) for p in trimmed]

    def _tokenize(self, text: str) -> set[str]:
        tokens = set(re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]+", text.lower()))
        # Add Chinese single-char tokens to improve retrieval for short Chinese phrases.
        for ch in text:
            if "\u4e00" <= ch <= "\u9fff":
                tokens.add(ch)
        return tokens

    def _score_chunk(self, query_tokens: set[str], chunk: str) -> int:
        chunk_tokens = self._tokenize(chunk)
        return len(query_tokens.intersection(chunk_tokens))

    def _request_openai_compatible(
        self,
        api_base: str,
        api_key: str,
        model: str,
        prompt: str,
    ) -> str:
        base = api_base.rstrip("/")
        url = f"{base}/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        req = request.Request(url, data=data, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=25) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
                obj = json.loads(body)
                return str(obj["choices"][0]["message"]["content"])
        except Exception:
            return ""

    def _parse_llm_json(self, content: str) -> dict | None:
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    obj = json.loads(text[start : end + 1])
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
            return None
