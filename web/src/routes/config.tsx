import { useQuery, useQueryClient } from '@tanstack/react-query'
import {
  DatabaseBackup,
  Gauge,
  KeyRound,
  ListOrdered,
  RefreshCw,
  RotateCcw,
  Save,
  Settings2,
  ShieldCheck,
  TriangleAlert,
} from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { toast } from 'sonner'

import { ConfigDiffDialog } from '@/components/config/diff-dialog'
import {
  FirstRunGuide,
  ModelApiCard,
  ModelApiSummary,
  type ModelApiDraft,
} from '@/components/config/model-api-card'
import { PageHeader } from '@/components/layout/page-header'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Empty } from '@/components/ui/empty'
import { Skeleton } from '@/components/ui/skeleton'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { TitledCard } from '@/components/ui/titled-card'
import {
  ApiError,
  getConfig,
  getProviders,
  listConfigBackups,
  revertConfig,
  saveConfig,
  type ConfigBackup,
  type ConfigDiffRow,
  type ConfigPatch,
  type ConfigResponse,
  type EngineNumberKey,
  type LlmConfig,
  type ProviderPreset,
} from '@/lib/api'
import { syncedRoutingForModel } from '@/lib/model-routing'
import { cn } from '@/lib/utils'

const INPUT_CLASS =
  'h-9 w-full rounded-md border border-border bg-background px-3 text-sm text-foreground placeholder:text-muted-foreground/70'

const ROLE_ORDER = [
  'agent',
  'agent_strong',
  'planning',
  'planning_expert',
  'intent',
  'task_refine',
  'recovery',
  'recovery_expert',
  'summarize',
  'verify',
] as const

const ROLE_LABEL: Record<string, string> = {
  agent: '主代理',
  agent_strong: '强模型代理',
  planning: '规划',
  planning_expert: '规划专家',
  intent: '意图识别',
  task_refine: '任务细化',
  recovery: '错误恢复',
  recovery_expert: '恢复专家',
  summarize: '总结',
  verify: '验收',
}

interface EngineField {
  key: EngineNumberKey
  label: string
  hint: string
  min: number
}

const ENGINE_FIELDS: EngineField[] = [
  { key: 'max_turns', label: '最大轮次', hint: '一个 run 最多执行多少轮工具调用', min: 1 },
  { key: 'max_repairs', label: '最大修复次数', hint: '验收失败后允许自动修复的轮数', min: 0 },
  {
    key: 'context_soft_limit_tokens',
    label: '上下文软上限（token）',
    hint: '超过后触发上下文压缩 / 摘要',
    min: 1000,
  },
  {
    key: 'kernel_idle_timeout_seconds',
    label: '内核空闲超时（秒）',
    hint: 'ArcPy 内核空闲多久回收',
    min: 30,
  },
  {
    key: 'kernel_exec_timeout_seconds',
    label: '内核执行超时（秒）',
    hint: '单次工具执行的最长等待',
    min: 30,
  },
]

const ENGINE_KEYS = ENGINE_FIELDS.map((field) => field.key)

interface Draft {
  provider: string
  model: string
  apiBase: string
  timeout: string
  routing: Record<string, string>
  fallback: string
  engine: Record<EngineNumberKey, string>
  temperature: string
  maxTokens: string
  apiKey: string
}

function rolesOf(config: LlmConfig): string[] {
  const extra = Object.keys(config.routing_rules ?? {}).filter(
    (role) => !(ROLE_ORDER as readonly string[]).includes(role)
  )
  return [...ROLE_ORDER, ...extra]
}

function modelsText(models: string[] | undefined): string {
  return (models ?? []).join(', ')
}

/** 中英文逗号 / 换行都当分隔符：用户从文档里粘贴也不会变成一个模型名。 */
function splitModels(text: string): string[] {
  return text
    .split(/[,，\n]/)
    .map((item) => item.trim())
    .filter(Boolean)
}

function numberText(value: unknown): string {
  return typeof value === 'number' && Number.isFinite(value) ? String(value) : ''
}

/** 与后端 `_require_api_base` 同口径：空串合法，否则必须是干净的 http(s) 地址。 */
function validateApiBase(value: string): string | null {
  if (!value) return null
  if (value.length > 500) return '「模型地址」太长（≤500 字符）'
  if (!/^https?:\/\//i.test(value)) return '「模型地址」必须以 http:// 或 https:// 开头'
  if (/\s/.test(value)) return '「模型地址」不能包含空格'
  if (/[\u0000-\u001f\u007f]/.test(value)) return '「模型地址」不能包含控制字符'
  if (value.includes('?')) return '「模型地址」不能带查询串（?...）'
  if (value.includes('#')) return '「模型地址」不能带片段（#...）'
  const authority = value.split('://')[1]?.split('/')[0] ?? ''
  if (!authority) return '「模型地址」缺少主机名'
  if (authority.includes('@')) return '「模型地址」不能包含账号密码（user:pass@）'
  return null
}

function draftFrom(config: LlmConfig): Draft {
  const routing: Record<string, string> = {}
  for (const role of rolesOf(config)) routing[role] = modelsText(config.routing_rules?.[role])
  const engine = {} as Record<EngineNumberKey, string>
  for (const key of ENGINE_KEYS) engine[key] = numberText(config.engine?.[key])
  return {
    provider: config.provider ?? '',
    model: config.model ?? '',
    apiBase: config.api_base ?? '',
    timeout: numberText(config.timeout),
    routing,
    fallback: modelsText(config.fallback_models),
    engine,
    temperature: numberText(config.temperature),
    maxTokens: numberText(config.max_tokens),
    apiKey: '',
  }
}

interface Edit {
  patch: ConfigPatch
  errors: string[]
  apiKey: string
  count: number
  dirty: boolean
}

/** 把当前草稿和磁盘上的配置对比，产出「只含改动」的 patch + 校验错误。 */
function buildEdit(draft: Draft | null, config: LlmConfig | undefined): Edit {
  const empty: Edit = { patch: {}, errors: [], apiKey: '', count: 0, dirty: false }
  if (!draft || !config) return empty

  const errors: string[] = []
  const patch: ConfigPatch = {}
  let count = 0

  // 模型与 API：provider 只是元信息，真正生效的是 api_base + model + api_key。
  const provider = draft.provider.trim()
  if (provider !== (config.provider ?? '')) {
    patch.provider = provider
    count += 1
  }

  const model = draft.model.trim()
  if (!model) {
    errors.push('「模型名 model」不能为空（可先点「从网关获取候选」）')
  } else if (model.length > 200) {
    errors.push('「模型名 model」太长（≤200 字符）')
  } else if (model !== (config.model ?? '')) {
    patch.model = model
    count += 1
  }

  const apiBase = draft.apiBase.trim()
  const apiBaseError = validateApiBase(apiBase)
  if (apiBaseError) {
    errors.push(apiBaseError)
  } else if (apiBase !== (config.api_base ?? '')) {
    patch.api_base = apiBase
    count += 1
  }

  const timeoutRaw = draft.timeout.trim()
  if (!timeoutRaw) {
    if (config.timeout !== undefined) errors.push('「请求超时 timeout」不能留空')
  } else {
    const timeout = Number(timeoutRaw)
    if (!Number.isFinite(timeout) || timeout < 5 || timeout > 3600) {
      errors.push('「请求超时 timeout」必须在 5 ~ 3600 秒之间')
    } else if (timeout !== config.timeout) {
      patch.timeout = timeout
      count += 1
    }
  }

  const routing: Record<string, string[]> = {}
  for (const role of rolesOf(config)) {
    const parsed = splitModels(draft.routing[role] ?? '')
    const original = config.routing_rules?.[role] ?? []
    // 空配置首次保存：没填的角色不写、也不拦（引擎会用顶层 model 兜底）。
    if (parsed.length === 0 && original.length === 0) continue
    if (parsed.length === 0) errors.push(`「${ROLE_LABEL[role] ?? role}」至少要有一个候选模型`)
    if (parsed.join('\u0000') !== original.join('\u0000')) {
      routing[role] = parsed
      count += 1
    }
  }
  if (Object.keys(routing).length > 0) patch.routing_rules = routing

  // 与后端 `apply_patch` 同口径：引擎里 routing_rules 排在 model 前面，只改模型名
  // 不同步路由的话 10 个角色仍会先试旧模型（DAIR-265）。上面显式改过的角色以草稿为准。
  if (patch.model) {
    const synced = syncedRoutingForModel({
      routingRules: config.routing_rules,
      roles: rolesOf(config),
      model: patch.model,
      explicit: patch.routing_rules,
    })
    if (Object.keys(synced).length > 0) {
      patch.routing_rules = { ...(patch.routing_rules ?? {}), ...synced }
      count += Object.keys(synced).length
    }
  }

  const fallback = splitModels(draft.fallback)
  if (fallback.join('\u0000') !== (config.fallback_models ?? []).join('\u0000')) {
    patch.fallback_models = fallback
    count += 1
  }

  const engine: Partial<Record<EngineNumberKey, number>> = {}
  for (const field of ENGINE_FIELDS) {
    const raw = (draft.engine[field.key] ?? '').trim()
    if (!raw) {
      // 空配置没填就用引擎默认值；已有值的配置清空仍然报错。
      if (config.engine?.[field.key] === undefined) continue
      errors.push(`「${field.label}」不能留空`)
      continue
    }
    const value = Number(raw)
    if (!Number.isInteger(value)) {
      errors.push(`「${field.label}」必须是整数`)
      continue
    }
    if (value < field.min) {
      errors.push(`「${field.label}」不能小于 ${field.min}`)
      continue
    }
    if (value !== config.engine?.[field.key]) {
      engine[field.key] = value
      count += 1
    }
  }
  if (Object.keys(engine).length > 0) patch.engine = engine

  const temperatureRaw = draft.temperature.trim()
  if (!temperatureRaw) {
    if (config.temperature !== undefined) errors.push('「温度 temperature」不能留空')
  } else {
    const temperature = Number(temperatureRaw)
    if (!Number.isFinite(temperature) || temperature < 0 || temperature > 2) {
      errors.push('「温度 temperature」必须在 0 ~ 2 之间')
    } else if (temperature !== config.temperature) {
      patch.temperature = temperature
      count += 1
    }
  }

  const maxTokensRaw = draft.maxTokens.trim()
  if (!maxTokensRaw) {
    if (config.max_tokens !== undefined) errors.push('「最大输出 max_tokens」不能留空')
  } else {
    const maxTokens = Number(maxTokensRaw)
    if (!Number.isInteger(maxTokens) || maxTokens < 1) {
      errors.push('「最大输出 max_tokens」必须是正整数')
    } else if (maxTokens !== config.max_tokens) {
      patch.max_tokens = maxTokens
      count += 1
    }
  }

  const apiKey = draft.apiKey.trim()
  if (apiKey) {
    if (apiKey.includes('…') || apiKey.startsWith('***')) {
      errors.push('API Key 不能填脱敏值（sk-…后4位）；留空表示不修改')
    } else {
      count += 1
    }
  }

  return { patch, errors, apiKey, count, dirty: count > 0 }
}

/** 客户端 diff 预览：把 patch 叠到当前脱敏配置上，逐键对比。 */
function applyPatch(base: LlmConfig, patch: ConfigPatch, apiKey: string): LlmConfig {
  const next: LlmConfig = {
    ...base,
    routing_rules: { ...(base.routing_rules ?? {}) },
    engine: { ...(base.engine ?? {}) },
  }
  if (patch.routing_rules) next.routing_rules = { ...next.routing_rules, ...patch.routing_rules }
  if (patch.provider !== undefined) next.provider = patch.provider
  if (patch.model !== undefined) next.model = patch.model
  if (patch.api_base !== undefined) next.api_base = patch.api_base
  if (patch.timeout !== undefined) next.timeout = patch.timeout
  if (patch.fallback_models) next.fallback_models = patch.fallback_models
  if (patch.engine) next.engine = { ...next.engine, ...patch.engine }
  if (patch.temperature !== undefined) next.temperature = patch.temperature
  if (patch.max_tokens !== undefined) next.max_tokens = patch.max_tokens
  // 只把后 4 位带进预览，真 key 不留在 DOM 里。
  if (apiKey) next.api_key = `sk-…${apiKey.slice(-4)}`
  return next
}

function flatten(value: unknown, prefix = ''): Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    return prefix ? { [prefix]: value } : {}
  }
  const flat: Record<string, unknown> = {}
  for (const [key, item] of Object.entries(value as Record<string, unknown>)) {
    Object.assign(flat, flatten(item, prefix ? `${prefix}.${key}` : key))
  }
  return flat
}

function diffRows(before: LlmConfig, after: LlmConfig): ConfigDiffRow[] {
  const left = flatten(before)
  const right = flatten(after)
  const rows: ConfigDiffRow[] = []
  for (const path of [...new Set([...Object.keys(left), ...Object.keys(right)])].sort()) {
    const oldValue = left[path]
    const newValue = right[path]
    if (JSON.stringify(oldValue) === JSON.stringify(newValue)) continue
    rows.push({
      path,
      before: oldValue,
      after: newValue,
      secret: path.split('.').some((part) => part.toLowerCase().endsWith('api_key')),
    })
  }
  return rows
}

function formatBytes(size: number): string {
  if (size < 1024) return `${size} B`
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`
  return `${(size / 1024 / 1024).toFixed(1)} MB`
}

function formatBackupTime(value: string): string {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN', { hour12: false })
}

function ConfigSkeleton() {
  return (
    <div className="space-y-4">
      <Skeleton className="h-14 w-full" />
      <Skeleton className="h-10 w-72" />
      <Skeleton className="h-[420px] w-full" />
    </div>
  )
}

function BusyPill({ busy, runId }: { busy: boolean; runId: string | null }) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium',
        busy ? 'border-warning/40 text-warning' : 'border-success/40 text-success'
      )}
    >
      <span className="size-1.5 rounded-full bg-current" />
      {busy ? `运行中 · 不可写${runId ? `（${runId.slice(0, 8)}）` : ''}` : '空闲 · 可写'}
    </span>
  )
}

export function ConfigPage() {
  const queryClient = useQueryClient()
  const configQuery = useQuery({
    queryKey: ['config'],
    queryFn: getConfig,
    refetchInterval: 15000,
  })
  const backupsQuery = useQuery({
    queryKey: ['config-backups'],
    queryFn: listConfigBackups,
    refetchInterval: 30000,
  })
  const providersQuery = useQuery({
    queryKey: ['config-providers'],
    queryFn: getProviders,
    staleTime: 5 * 60_000,
    retry: false,
  })

  const data: ConfigResponse | undefined = configQuery.data
  const [draft, setDraft] = useState<Draft | null>(null)
  const [baseSha, setBaseSha] = useState('')
  const [stale, setStale] = useState(false)
  const [diffOpen, setDiffOpen] = useState(false)
  const [saving, setSaving] = useState(false)
  const [revertTarget, setRevertTarget] = useState<ConfigBackup | null>(null)
  const [reverting, setReverting] = useState(false)
  const dirtyRef = useRef(false)

  const edit = useMemo(() => buildEdit(draft, data), [draft, data])
  dirtyRef.current = edit.dirty

  useEffect(() => {
    if (!data) return
    if (!draft) {
      setDraft(draftFrom(data))
      setBaseSha(data.sha256)
      return
    }
    if (data.sha256 === baseSha) return
    if (dirtyRef.current) {
      setStale(true)
      return
    }
    setDraft(draftFrom(data))
    setBaseSha(data.sha256)
    setStale(false)
  }, [data, draft, baseSha])

  const busy = Boolean(data?._meta.busy)
  const roleList = data ? rolesOf(data) : [...ROLE_ORDER]
  const firstRun = data
    ? !data._meta.active.config_exists || data._meta.active.model === ''
    : false

  // 预设表读不到时至少保住当前值，否则下拉一空就会把已有 provider 写没了。
  const providers: ProviderPreset[] = useMemo(() => {
    const items = providersQuery.data?.items ?? []
    if (items.length > 0) return items
    const current = draft?.provider
    return current
      ? [{ id: current, name: current, default_base: '', key_required: false, hint: '' }]
      : []
  }, [providersQuery.data, draft?.provider])

  const changeProvider = (id: string) => {
    setDraft((current) => {
      if (!current) return current
      const prev = providers.find((item) => item.id === current.provider)
      const next = providers.find((item) => item.id === id)
      const base = current.apiBase.trim()
      // 只在地址为空、或还是上一个预设的默认值时自动填；用户自己填的反代地址不覆盖。
      const autoFill = !base || base === (prev?.default_base ?? '')
      return {
        ...current,
        provider: id,
        apiBase: autoFill ? (next?.default_base ?? '') : current.apiBase,
      }
    })
  }

  const patchModelApi = (
    next: Partial<Pick<ModelApiDraft, 'model' | 'apiBase' | 'timeout' | 'apiKey'>>
  ) => setDraft((current) => (current ? { ...current, ...next } : current))

  const previewRows = useMemo(
    () => (data ? diffRows(data, applyPatch(data, edit.patch, edit.apiKey)) : []),
    [data, edit]
  )

  const reload = () => {
    if (!data) return
    setDraft(draftFrom(data))
    setBaseSha(data.sha256)
    setStale(false)
    toast.success('已重新读取磁盘上的配置')
  }

  const reset = () => {
    if (!data) return
    setDraft(draftFrom(data))
    setStale(false)
  }

  const save = async () => {
    if (!data || !draft) return
    if (edit.errors.length > 0) {
      toast.error(edit.errors[0])
      return
    }
    setSaving(true)
    try {
      const result = await saveConfig(edit.patch, data.sha256, edit.apiKey)
      setDiffOpen(false)
      // 先拉回新配置再换 draft/baseSha：否则中间那一帧会让 effect 误判成
      // "别处改了配置" 而弹出 stale 警告。
      const fresh = await configQuery.refetch()
      if (fresh.data) {
        setDraft(draftFrom(fresh.data))
        setBaseSha(fresh.data.sha256)
      } else {
        setBaseSha(result.sha256)
        setDraft({ ...draft, apiKey: '' })
      }
      setStale(false)
      toast.success(result.diff.length > 0 ? `已保存 ${result.diff.length} 处改动` : '没有需要保存的改动', {
        description: result.backup ? `已备份 ${result.backup}` : undefined,
      })
      await queryClient.invalidateQueries({ queryKey: ['config-backups'] })
    } catch (error) {
      const message = error instanceof ApiError ? error.message : '保存失败'
      toast.error(message)
      if (error instanceof ApiError && error.status === 409) {
        void configQuery.refetch()
      }
    } finally {
      setSaving(false)
    }
  }

  const revert = async () => {
    if (!revertTarget) return
    setReverting(true)
    try {
      const result = await revertConfig(revertTarget.name)
      setRevertTarget(null)
      // 还原是用户在本页发起的：直接把草稿对齐到还原后的文件，
      // 否则页面会把它当成“别人改了配置”而弹 stale 警告。
      const fresh = await configQuery.refetch()
      if (fresh.data) {
        setDraft(draftFrom(fresh.data))
        setBaseSha(fresh.data.sha256)
      }
      setStale(false)
      toast.success(`已从 ${result.restored_from} 还原`, {
        description: result.backup ? `还原前的文件也留了备份 ${result.backup}` : undefined,
      })
      await queryClient.invalidateQueries({ queryKey: ['config-backups'] })
    } catch (error) {
      toast.error(error instanceof ApiError ? error.message : '还原失败')
    } finally {
      setReverting(false)
    }
  }

  return (
    <>
      <PageHeader
        title="配置模型"
        description="10 个角色的模型路由、引擎预算与回退策略；写入前自动备份，密钥只显示脱敏值。"
        actions={
          <Button
            variant="outline"
            size="sm"
            onClick={() => configQuery.refetch()}
            disabled={configQuery.isFetching}
          >
            <RefreshCw className={cn('size-3.5', configQuery.isFetching && 'animate-spin')} />
            重新读取
          </Button>
        }
      />

      {configQuery.isLoading ? (
        <ConfigSkeleton />
      ) : configQuery.isError || !data || !draft ? (
        <Empty
          icon={Settings2}
          title="配置读不到"
          description={
            configQuery.error instanceof ApiError
              ? configQuery.error.message
              : 'config/llm_config.json 不存在或不是合法 JSON。'
          }
          action={
            <Button size="sm" onClick={() => configQuery.refetch()}>
              重试
            </Button>
          }
        />
      ) : (
        <>
          {firstRun ? (
            <FirstRunGuide config={data} provider={draft.provider} apiKey={draft.apiKey} />
          ) : null}

          <ModelApiSummary
            config={data}
            providers={providers}
            draft={{
              provider: draft.provider,
              model: draft.model,
              apiBase: draft.apiBase,
              timeout: draft.timeout,
              apiKey: draft.apiKey,
            }}
          />

          <section className="mb-4 rounded-lg border border-border bg-card px-4 py-3">
            <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
              <span className="inline-flex items-center gap-2 text-xs text-muted-foreground">
                <KeyRound className="size-3.5" />
                API Key
                <code className="rounded-sm bg-muted px-1.5 py-0.5 font-mono text-[11px] text-foreground">
                  {data.api_key || '未配置'}
                </code>
              </span>
              <span className="text-[11px] text-muted-foreground">
                sha256{' '}
                <code className="font-mono text-foreground">
                  {data.sha256 ? `${data.sha256.slice(0, 12)}…` : '（空配置）'}
                </code>
              </span>
              <span className="text-[11px] text-muted-foreground">
                文件 <code className="font-mono text-foreground">{data._meta.path}</code>
              </span>
              <span className="ml-auto">
                <BusyPill busy={busy} runId={data._meta.busy_run} />
              </span>
            </div>
            {busy ? (
              <p className="mt-2 border-t border-border pt-2 text-[11px] text-muted-foreground">
                有 run 正在运行：保存 / 还原会被拒绝（409），配置只对下一个 run 生效。
              </p>
            ) : null}
          </section>

          {stale ? (
            <div className="mb-4 flex flex-wrap items-center gap-3 rounded-lg border border-warning/40 bg-warning/10 px-4 py-3 text-xs text-warning">
              <TriangleAlert className="size-4" />
              磁盘上的配置已在别处被修改；当前编辑基于旧版本，保存会因 sha256 不匹配被拒绝。
              <Button variant="outline" size="sm" className="ml-auto" onClick={reload}>
                放弃本地改动并重新载入
              </Button>
            </div>
          ) : null}

          <div className="mb-4">
            <ModelApiCard
              config={data}
              providers={providers}
              providersError={
                providersQuery.isError
                  ? '供应商预设读不到（接口不可用），下拉里只保留当前值。'
                  : undefined
              }
              draft={{
                provider: draft.provider,
                model: draft.model,
                apiBase: draft.apiBase,
                timeout: draft.timeout,
                apiKey: draft.apiKey,
              }}
              onChange={patchModelApi}
              onProviderChange={changeProvider}
            />
          </div>

          <Tabs defaultValue="routing">
            <TabsList>
              <TabsTrigger value="routing">模型路由</TabsTrigger>
              <TabsTrigger value="engine">引擎预算</TabsTrigger>
              <TabsTrigger value="fallback">回退与容错</TabsTrigger>
              <TabsTrigger value="backups">备份</TabsTrigger>
            </TabsList>

            <TabsContent value="routing">
              <TitledCard
                title="模型路由"
                icon={ListOrdered}
                meta={`${roleList.length} 个角色`}
                bodyClassName="px-0 py-0"
              >
                <p className="px-4 py-3 text-xs text-muted-foreground">
                  候选顺序从左到右依次尝试，第一个即主模型；逗号分隔（中英文逗号均可）。
                </p>
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[44rem] text-sm">
                    <thead>
                      <tr className="border-y border-border text-xs text-muted-foreground">
                        <th className="w-[13rem] px-4 py-2 text-left font-medium">角色</th>
                        <th className="px-3 py-2 text-left font-medium">候选顺序（可直接编辑）</th>
                        <th className="w-[11rem] px-4 py-2 text-left font-medium">主模型</th>
                      </tr>
                    </thead>
                    <tbody>
                      {roleList.map((role) => {
                        const models = splitModels(draft.routing[role] ?? '')
                        return (
                          <tr key={role} className="border-b border-border last:border-0">
                            <td className="px-4 py-2.5 align-middle">
                              <div className="text-xs font-medium text-foreground">
                                {ROLE_LABEL[role] ?? role}
                              </div>
                              <div className="font-mono text-[11px] text-muted-foreground">
                                {role}
                              </div>
                            </td>
                            <td className="px-3 py-2 align-middle">
                              <input
                                className={INPUT_CLASS}
                                value={draft.routing[role] ?? ''}
                                placeholder="model-a, model-b"
                                aria-label={`${ROLE_LABEL[role] ?? role} 的候选模型`}
                                onChange={(event) =>
                                  setDraft({
                                    ...draft,
                                    routing: { ...draft.routing, [role]: event.target.value },
                                  })
                                }
                              />
                            </td>
                            <td className="px-4 py-2.5 align-middle">
                              {models[0] ? (
                                <div className="min-w-0">
                                  <span className="inline-block max-w-full truncate rounded-full border border-border bg-background px-2 py-0.5 text-[11px] text-foreground">
                                    {models[0]}
                                  </span>
                                  <div className="mt-1 text-[11px] text-muted-foreground">
                                    {models.length} 个候选
                                  </div>
                                </div>
                              ) : (
                                <span className="text-[11px] text-destructive">未填</span>
                              )}
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </TitledCard>
            </TabsContent>

            <TabsContent value="engine">
              <TitledCard title="引擎预算" icon={Gauge} meta="改完只对下一个 run 生效">
                <div className="divide-y divide-border">
                  {ENGINE_FIELDS.map((field) => (
                    <div
                      key={field.key}
                      className="flex flex-wrap items-center gap-3 py-3 first:pt-0 last:pb-0"
                    >
                      <div className="min-w-0 flex-1">
                        <label
                          className="text-xs font-medium text-foreground"
                          htmlFor={`engine-${field.key}`}
                        >
                          {field.label}
                        </label>
                        <div className="font-mono text-[11px] text-muted-foreground">
                          engine.{field.key}
                        </div>
                        <div className="text-[11px] text-muted-foreground">{field.hint}</div>
                      </div>
                      <input
                        id={`engine-${field.key}`}
                        className={cn(INPUT_CLASS, 'max-w-[11rem] text-right tabular-nums')}
                        inputMode="numeric"
                        value={draft.engine[field.key] ?? ''}
                        onChange={(event) =>
                          setDraft({
                            ...draft,
                            engine: { ...draft.engine, [field.key]: event.target.value },
                          })
                        }
                      />
                    </div>
                  ))}
                </div>
              </TitledCard>
            </TabsContent>

            <TabsContent value="fallback">
              <div className="space-y-4">
                <TitledCard title="回退与容错" icon={ShieldCheck}>
                  <div className="space-y-4">
                    <div>
                      <label
                        className="text-xs font-medium text-foreground"
                        htmlFor="fallback-models"
                      >
                        回退模型 fallback_models
                      </label>
                      <p className="mb-1.5 text-[11px] text-muted-foreground">
                        主模型全部失败时按顺序尝试；留空表示不设回退。
                      </p>
                      <input
                        id="fallback-models"
                        className={INPUT_CLASS}
                        value={draft.fallback}
                        placeholder="model-b, model-c"
                        onChange={(event) => setDraft({ ...draft, fallback: event.target.value })}
                      />
                    </div>
                    <div className="grid gap-4 sm:grid-cols-2">                      <div>
                        <label className="text-xs font-medium text-foreground" htmlFor="temperature">
                          温度 temperature
                        </label>
                        <p className="mb-1.5 text-[11px] text-muted-foreground">0 ~ 2，越低越稳定。</p>
                        <input
                          id="temperature"
                          className={cn(INPUT_CLASS, 'tabular-nums')}
                          inputMode="decimal"
                          value={draft.temperature}
                          onChange={(event) =>
                            setDraft({ ...draft, temperature: event.target.value })
                          }
                        />
                      </div>
                      <div>
                        <label className="text-xs font-medium text-foreground" htmlFor="max-tokens">
                          最大输出 max_tokens
                        </label>
                        <p className="mb-1.5 text-[11px] text-muted-foreground">
                          单次回复的 token 上限。
                        </p>
                        <input
                          id="max-tokens"
                          className={cn(INPUT_CLASS, 'tabular-nums')}
                          inputMode="numeric"
                          value={draft.maxTokens}
                          onChange={(event) => setDraft({ ...draft, maxTokens: event.target.value })}
                        />
                      </div>
                    </div>
                    <p className="border-t border-border pt-4 text-[11px] text-muted-foreground">
                      API Key 已移到页面顶部的「模型与 API」卡（与地址、模型名放在一起）。
                    </p>
                  </div>
                </TitledCard>
              </div>
            </TabsContent>

            <TabsContent value="backups">
              <TitledCard
                title="备份"
                icon={DatabaseBackup}
                meta={`${backupsQuery.data?.count ?? 0} 份 · ${data._meta.backup_dir}`}
                actions={
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => backupsQuery.refetch()}
                    disabled={backupsQuery.isFetching}
                  >
                    <RefreshCw
                      className={cn('size-3.5', backupsQuery.isFetching && 'animate-spin')}
                    />
                    刷新
                  </Button>
                }
                bodyClassName="px-0 py-0"
              >
                {backupsQuery.isLoading ? (
                  <div className="space-y-2 px-4 py-4">
                    <Skeleton className="h-4 w-2/3" />
                    <Skeleton className="h-4 w-1/2" />
                  </div>
                ) : backupsQuery.isError ? (
                  <p className="px-4 py-6 text-xs text-destructive">
                    {backupsQuery.error instanceof ApiError
                      ? backupsQuery.error.message
                      : '备份列表读不到'}
                  </p>
                ) : (backupsQuery.data?.items.length ?? 0) === 0 ? (
                  <Empty
                    className="min-h-[220px] rounded-none border-0 bg-transparent"
                    icon={DatabaseBackup}
                    title="还没有备份"
                    description="每次保存配置前会自动生成 config/backups/llm_config-<时间戳>.json。"
                  />
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full min-w-[40rem] text-sm">
                      <thead>
                        <tr className="border-b border-border text-xs text-muted-foreground">
                          <th className="px-4 py-2 text-left font-medium">时间</th>
                          <th className="px-3 py-2 text-left font-medium">文件</th>
                          <th className="px-3 py-2 text-right font-medium">大小</th>
                          <th className="px-4 py-2 text-right font-medium">操作</th>
                        </tr>
                      </thead>
                      <tbody>
                        {backupsQuery.data?.items.map((item) => (
                          <tr key={item.name} className="border-b border-border last:border-0">
                            <td className="px-4 py-2.5 text-xs tabular-nums text-foreground">
                              {formatBackupTime(item.created_at)}
                            </td>
                            <td className="px-3 py-2.5">
                              <div className="font-mono text-[11px] text-foreground">{item.name}</div>
                              <div className="font-mono text-[11px] text-muted-foreground">
                                {item.sha256.slice(0, 12)}…
                              </div>
                            </td>
                            <td className="px-3 py-2.5 text-right text-xs tabular-nums text-muted-foreground">
                              {formatBytes(item.size_bytes)}
                            </td>
                            <td className="px-4 py-2.5 text-right">
                              <Button
                                variant="outline"
                                size="sm"
                                disabled={busy || reverting}
                                onClick={() => setRevertTarget(item)}
                              >
                                <RotateCcw className="size-3.5" />
                                还原
                              </Button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </TitledCard>
            </TabsContent>
          </Tabs>

          <div className="sticky bottom-4 z-10 mt-4 flex flex-wrap items-center gap-3 rounded-lg border border-border bg-card/95 px-4 py-3 backdrop-blur">
            <div className="min-w-0 text-xs">
              {edit.dirty ? (
                <>
                  <span className="font-medium text-foreground">{edit.count} 项未保存</span>
                  <span className="text-muted-foreground"> · 保存前会预览 diff 并二次确认</span>
                </>
              ) : null}
              {edit.errors.length > 0 ? (
                <div className={cn('text-[11px] text-destructive', edit.dirty && 'mt-0.5')}>
                  还需补全：{edit.errors[0]}
                </div>
              ) : !edit.dirty ? (
                <span className="text-muted-foreground">没有未保存的改动</span>
              ) : null}
            </div>
            <div className="ml-auto flex items-center gap-2">
              <Button variant="ghost" size="sm" onClick={reset} disabled={!edit.dirty || saving}>
                放弃改动
              </Button>
              <Button
                size="sm"
                onClick={() => setDiffOpen(true)}
                disabled={!edit.dirty || edit.errors.length > 0 || saving || busy}
                title={busy ? '有 run 正在运行，暂时不可写' : undefined}
              >
                <Save className="size-3.5" />
                保存…
              </Button>
            </div>
          </div>

          <ConfigDiffDialog
            open={diffOpen}
            onOpenChange={setDiffOpen}
            rows={previewRows}
            apiKeyChanged={Boolean(edit.apiKey)}
            saving={saving}
            onConfirm={save}
          />

          <Dialog
            open={revertTarget !== null}
            onOpenChange={(open) => {
              if (!open) setRevertTarget(null)
            }}
          >
            <DialogContent>
              <DialogHeader>
                <DialogTitle>从这个备份还原？</DialogTitle>
                <DialogDescription>
                  会用 <span className="font-mono">{revertTarget?.name}</span>{' '}
                  覆盖当前 config/llm_config.json；还原前当前文件也会自动备份，可再还原回来。
                  该操作只对下一个 run 生效。
                </DialogDescription>
              </DialogHeader>
              <DialogFooter>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setRevertTarget(null)}
                  disabled={reverting}
                >
                  取消
                </Button>
                <Button size="sm" onClick={revert} disabled={reverting || busy}>
                  {reverting ? '还原中…' : '确认还原'}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </>
      )}
    </>
  )
}
