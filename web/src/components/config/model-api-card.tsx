import { Check, ChevronDown, Eye, EyeOff, LoaderCircle, Plug, RotateCcw, WandSparkles } from 'lucide-react'
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Select } from '@/components/ui/select'
import { TitledCard } from '@/components/ui/titled-card'
import {
  ApiError,
  testConnection,
  type ConfigResponse,
  type ConfigTestResult,
  type ProviderPreset,
} from '@/lib/api'
import { cn } from '@/lib/utils'

const INPUT_CLASS =
  'h-9 w-full rounded-md border border-border bg-background px-3 text-sm text-foreground outline-none placeholder:text-muted-foreground/70 focus:border-primary/50'

/** 草稿里与「模型与 API」相关的五个字段（都是字符串，保存时才转数字）。 */
export interface ModelApiDraft {
  provider: string
  model: string
  apiBase: string
  timeout: string
  apiKey: string
}

interface TestBody {
  provider?: string
  api_base: string
  model?: string
  api_key?: string
}

/** 测试连接的最小状态机：同一份 API 的两处入口（摘要卡 / 卡片内）各自持有。 */
function useConnectionTest() {
  const [testing, setTesting] = useState(false)
  const [result, setResult] = useState<ConfigTestResult | null>(null)

  const run = async (body: TestBody) => {
    setTesting(true)
    setResult(null)
    try {
      setResult(await testConnection(body))
    } catch (error) {
      // 网络/鉴权失败也落到同一条展示路径上，不让按钮卡在「测试中」。
      setResult({
        ok: false,
        latency_ms: 0,
        models: [],
        error: error instanceof ApiError ? error.message : '测试请求没有发出去',
      })
    } finally {
      setTesting(false)
    }
  }

  return { testing, result, run }
}

function TestResultLine({ result }: { result: ConfigTestResult }) {
  if (result.ok) {
    return (
      <span className="text-[11px] text-success">
        延迟 {result.latency_ms} ms · 候选 {result.models.length} 个
      </span>
    )
  }
  return (
    <span className="text-[11px] text-destructive" title={result.error}>
      失败 · {result.latency_ms} ms · {result.error || '未知错误'}
    </span>
  )
}

/** 顶部「当前生效」摘要：只读磁盘上真正生效的 provider/model/api_base/key 尾部。 */
export function ModelApiSummary({
  config,
  providers,
  draft,
}: {
  config: ConfigResponse
  providers: ProviderPreset[]
  /** 用草稿值测试（可以先测再保存）。 */
  draft: ModelApiDraft
}) {
  const test = useConnectionTest()
  const active = config._meta.active
  const providerName =
    providers.find((item) => item.id === active.provider)?.name || active.provider || '未选供应商'

  const runTest = () => {
    void test.run({
      provider: draft.provider || undefined,
      api_base: draft.apiBase.trim(),
      model: draft.model.trim() || undefined,
      api_key: draft.apiKey.trim() || undefined,
    })
  }

  return (
    <section className="mb-4 rounded-lg border border-border bg-card px-4 py-3">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <span className="inline-flex items-center gap-2 text-xs font-medium text-foreground">
          <Plug className="size-3.5 text-primary" />
          当前生效
        </span>
        <code className="rounded-sm bg-muted px-1.5 py-0.5 font-mono text-[11px] text-foreground">
          {providerName}
        </code>
        <span className="text-[11px] text-muted-foreground">·</span>
        <code className="rounded-sm bg-muted px-1.5 py-0.5 font-mono text-[11px] text-foreground">
          {active.model || '未配置模型'}
        </code>
        <span className="text-[11px] text-muted-foreground">·</span>
        <code className="max-w-[22rem] truncate rounded-sm bg-muted px-1.5 py-0.5 font-mono text-[11px] text-foreground">
          {active.api_base || (active.config_exists ? '官方默认地址' : '未配置地址')}
        </code>
        <span className="text-[11px] text-muted-foreground">·</span>
        <span className="inline-flex items-center gap-1 text-[11px] text-muted-foreground">
          key
          <code className="font-mono text-foreground">
            {active.key_tail ? `sk-…${active.key_tail}` : '未配置'}
          </code>
        </span>
        <span className="ml-auto flex items-center gap-2">
          {test.result ? <TestResultLine result={test.result} /> : null}
          <Button variant="outline" size="sm" onClick={runTest} disabled={test.testing}>
            {test.testing ? (
              <LoaderCircle className="size-3.5 animate-spin" />
            ) : (
              <Plug className="size-3.5" />
            )}
            {test.testing ? '测试中…' : '测试连接'}
          </Button>
        </span>
      </div>
    </section>
  )
}

/** 首次引导：三步走，按磁盘现状点亮已完成项。 */
export function FirstRunGuide({
  config,
  provider,
  apiKey,
}: {
  config: ConfigResponse
  /** 还没保存的供应商选择（选中即算第一步做完）。 */
  provider: string
  /** 还没保存的 key 输入（填了也算第二步做完）。 */
  apiKey: string
}) {
  const active = config._meta.active
  const steps = [
    {
      title: '选供应商',
      hint: '反代 / 自建网关选「自定义」',
      done: provider !== '' || active.provider !== '',
    },
    {
      title: '填 API Key',
      hint: 'Ollama 本机可不填',
      done: active.key_tail !== '' || apiKey.trim() !== '',
    },
    {
      title: '测试连接并保存',
      hint: '保存前会预览 diff',
      done: active.config_exists && active.model !== '',
    },
  ]

  return (
    <section className="mb-4 rounded-lg border border-primary/30 bg-accent/40 px-4 py-3">
      <div className="flex items-center gap-2 text-xs font-medium text-foreground">
        <WandSparkles className="size-3.5 text-primary" />
        首次配置：三步把模型接上
      </div>
      <ol className="mt-2 grid gap-2 sm:grid-cols-3">
        {steps.map((step, index) => (
          <li
            key={step.title}
            className="flex items-start gap-2 rounded-md border border-border bg-card px-3 py-2"
          >
            <span
              className={cn(
                'mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full text-[11px] font-medium',
                step.done ? 'bg-success/15 text-success' : 'bg-muted text-muted-foreground'
              )}
            >
              {step.done ? <Check className="size-3" /> : index + 1}
            </span>
            <div className="min-w-0">
              <div className="text-xs font-medium text-foreground">{step.title}</div>
              <div className="text-[11px] text-muted-foreground">{step.hint}</div>
            </div>
          </li>
        ))}
      </ol>
    </section>
  )
}

export interface ModelApiCardProps {
  config: ConfigResponse
  providers: ProviderPreset[]
  /** providers 接口失败时的提示（下拉退化成只有当前值）。 */
  providersError?: string
  draft: ModelApiDraft
  onChange: (next: Partial<Pick<ModelApiDraft, 'model' | 'apiBase' | 'timeout' | 'apiKey'>>) => void
  onProviderChange: (id: string) => void
}

/**
 * 「模型与 API」表单卡：供应商 → 地址 → 模型名 → key，带测试连接与候选模型。
 *
 * 地址/模型名/key 只改本地草稿，保存仍走页面底部的 diff 确认；卡内不直接写盘。
 */
export function ModelApiCard({
  config,
  providers,
  providersError,
  draft,
  onChange,
  onProviderChange,
}: ModelApiCardProps) {
  // 首次配置时展开（引导指向本卡）；之后用户可自行收起。
  const [open, setOpen] = useState(true)
  const [showKey, setShowKey] = useState(false)
  const test = useConnectionTest()

  const selected = providers.find((item) => item.id === draft.provider)
  const active = config._meta.active
  const keyPlaceholder = active.key_tail
    ? `留空 = 不修改（当前 sk-…${active.key_tail}）`
    : '留空 = 不修改（当前未配置 key）'

  const runTest = () => {
    void test.run({
      provider: draft.provider || undefined,
      api_base: draft.apiBase.trim(),
      model: draft.model.trim() || undefined,
      api_key: draft.apiKey.trim() || undefined,
    })
  }

  return (
    <TitledCard
      title="模型与 API"
      icon={Plug}
      meta="保存后只对下一个 run 生效"
      actions={
        <Button variant="ghost" size="sm" onClick={() => setOpen(!open)}>
          <ChevronDown className={cn('size-3.5 transition-transform', open && 'rotate-180')} />
          {open ? '收起' : '展开'}
        </Button>
      }
    >
      {!open ? (
        <p className="text-[11px] text-muted-foreground">
          已收起。当前 {selected?.name || draft.provider || '未选供应商'} ·{' '}
          <span className="font-mono">{draft.model || '未填模型名'}</span>
        </p>
      ) : (
        <div className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <span className="text-xs font-medium text-foreground">供应商 provider</span>
              <p className="mb-1.5 text-[11px] text-muted-foreground">
                预设只决定默认地址与提示，真正生效的是下面的地址、模型名和 key。
              </p>
              <Select
                aria-label="供应商"
                className={cn(INPUT_CLASS, 'justify-between gap-2')}
                value={draft.provider}
                onValueChange={onProviderChange}
                options={providers.map((item) => ({ value: item.id, label: item.name }))}
                placeholder="选择供应商"
              />
              {selected ? (
                <p className="mt-1 text-[11px] text-muted-foreground">{selected.hint}</p>
              ) : null}
              {providersError ? (
                <p className="mt-1 text-[11px] text-destructive">{providersError}</p>
              ) : null}
            </div>
            <div>
              <label className="text-xs font-medium text-foreground" htmlFor="timeout-input">
                请求超时 timeout（秒）
              </label>
              <p className="mb-1.5 text-[11px] text-muted-foreground">5 ~ 3600，长任务可放大。</p>
              <input
                id="timeout-input"
                className={cn(INPUT_CLASS, 'tabular-nums')}
                inputMode="numeric"
                value={draft.timeout}
                placeholder="如 900"
                onChange={(event) => onChange({ timeout: event.target.value })}
              />
            </div>
          </div>

          <div>
            <label className="text-xs font-medium text-foreground" htmlFor="api-base-input">
              模型地址 api_base
            </label>
            <p className="mb-1.5 text-[11px] text-muted-foreground">
              留空 = 用官方默认地址；反代通常以 <span className="font-mono">/v1</span> 结尾，不能带
              <span className="font-mono"> ?query</span>。
            </p>
            <div className="flex items-center gap-2">
              <input
                id="api-base-input"
                className={cn(INPUT_CLASS, 'font-mono')}
                value={draft.apiBase}
                placeholder="http://localhost:8789/v1"
                onChange={(event) => onChange({ apiBase: event.target.value })}
              />
              <Button
                variant="outline"
                size="sm"
                className="shrink-0"
                disabled={!selected?.default_base || draft.apiBase === selected.default_base}
                onClick={() => onChange({ apiBase: selected?.default_base ?? '' })}
                title="填回当前供应商的默认地址"
              >
                <RotateCcw className="size-3.5" />
                恢复默认
              </Button>
            </div>
          </div>

          <div>
            <label className="text-xs font-medium text-foreground" htmlFor="model-input">
              模型名 model
            </label>
            <p className="mb-1.5 text-[11px] text-muted-foreground">
              网关上的真实模型 id。保存时，下方「模型路由」里没单独改过的角色会一起指到新模型
              （路由在引擎里优先于模型名，与 CLI <span className="font-mono">/model</span> 一致）。
            </p>
            <div className="flex items-center gap-2">
              <input
                id="model-input"
                className={cn(INPUT_CLASS, 'font-mono')}
                value={draft.model}
                placeholder="如 deepseek-v4.1-flash"
                onChange={(event) => onChange({ model: event.target.value })}
              />
              <Button
                variant="outline"
                size="sm"
                className="shrink-0"
                onClick={runTest}
                disabled={test.testing}
              >
                {test.testing ? (
                  <LoaderCircle className="size-3.5 animate-spin" />
                ) : (
                  <WandSparkles className="size-3.5" />
                )}
                {test.testing ? '探测中…' : '从网关获取候选'}
              </Button>
            </div>
            {test.result ? (
              <div className="mt-1.5 space-y-1.5">
                <TestResultLine result={test.result} />
                {test.result.ok && test.result.models.length > 0 ? (
                  <div className="flex flex-wrap gap-1.5">
                    {test.result.models.map((name) => (
                      <button
                        key={name}
                        type="button"
                        onClick={() => onChange({ model: name })}
                        className={cn(
                          'max-w-full truncate rounded-full border px-2 py-0.5 font-mono text-[11px] transition-colors',
                          name === draft.model
                            ? 'border-primary/50 bg-accent text-foreground'
                            : 'border-border bg-background text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                        )}
                        title={`填入 ${name}`}
                      >
                        {name}
                      </button>
                    ))}
                  </div>
                ) : null}
              </div>
            ) : null}
          </div>

          <div>
            <label className="text-xs font-medium text-foreground" htmlFor="api-key-input">
              API Key
            </label>
            <p className="mb-1.5 text-[11px] text-muted-foreground">
              {keyPlaceholder}。界面与日志都只显示后 4 位。
            </p>
            <div className="flex items-center gap-2">
              <input
                id="api-key-input"
                className={cn(INPUT_CLASS, 'font-mono')}
                type={showKey ? 'text' : 'password'}
                autoComplete="off"
                spellCheck={false}
                value={draft.apiKey}
                placeholder={keyPlaceholder}
                onChange={(event) => onChange({ apiKey: event.target.value })}
              />
              <Button
                variant="outline"
                size="sm"
                className="shrink-0"
                onClick={() => setShowKey(!showKey)}
                aria-label={showKey ? '隐藏 API Key' : '显示 API Key'}
                title={showKey ? '隐藏' : '显示'}
              >
                {showKey ? <EyeOff className="size-3.5" /> : <Eye className="size-3.5" />}
                {showKey ? '隐藏' : '显示'}
              </Button>
            </div>
          </div>
        </div>
      )}
    </TitledCard>
  )
}
