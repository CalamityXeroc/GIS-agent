/** 控制台 API 客户端：本机 token + 运行态读写。 */

const TOKEN_STORAGE_KEY = 'gis-agent-web-token'

/** 从 URL（`?token=`，`gis-agent web` 打印的那种）或 localStorage 取 token。 */
export function readToken(): string {
  if (typeof window === 'undefined') return ''
  try {
    const fromUrl = new URLSearchParams(window.location.search).get('token')
    if (fromUrl) {
      window.localStorage.setItem(TOKEN_STORAGE_KEY, fromUrl)
      return fromUrl
    }
    return window.localStorage.getItem(TOKEN_STORAGE_KEY) ?? ''
  } catch {
    return ''
  }
}

export function apiUrl(
  path: string,
  params: Record<string, string | number | undefined> = {}
): URL {
  const url = new URL(path, window.location.origin)
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) url.searchParams.set(key, String(value))
  }
  return url
}

/** SSE 用不了自定义请求头，token 走查询参数（服务端两种都认）。 */
export function runEventsUrl(runId: string, since = 0): string {
  const url = apiUrl(`/api/runs/${encodeURIComponent(runId)}/events`, { since })
  const token = readToken()
  if (token) url.searchParams.set('token', token)
  return url.toString()
}

export class ApiError extends Error {
  status: number
  /** 服务端 `detail` 原文（字符串，或预览超时这类结构化对象）。 */
  detail?: unknown

  constructor(status: number, message: string, detail?: unknown) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers)
  const token = readToken()
  if (token) headers.set('X-Web-Token', token)
  if (init.body && !headers.has('Content-Type')) headers.set('Content-Type', 'application/json')

  const response = await fetch(apiUrl(path), { ...init, headers })
  if (!response.ok) {
    let detail: string = response.statusText
    let payload: unknown
    try {
      const body = (await response.json()) as { detail?: unknown }
      payload = body.detail
      if (typeof body.detail === 'string') detail = body.detail
      else if (body.detail) detail = JSON.stringify(body.detail)
    } catch {
      /* 非 JSON 错误体：保留状态文本 */
    }
    throw new ApiError(response.status, detail, payload)
  }
  return (await response.json()) as T
}

export type RunStatus =
  | 'queued'
  | 'running'
  | 'awaiting_plan'
  | 'stopping'
  | 'succeeded'
  | 'failed'
  | 'stopped'
  | 'interrupted'

export interface PlanRow {
  label: string
  value: string
}

export interface PlanPayload {
  round?: number
  text?: string
  rows?: PlanRow[]
  goal?: string
  deadline?: string
}

export interface RunDetail {
  id: string
  goal: string
  status: RunStatus
  created_at: string
  started_at: string | null
  ended_at: string | null
  duration_s: number | null
  turns: number
  tokens: number
  repairs: number
  model: string
  current_tool: string
  verdict: string | null
  error: string
  resume_from: string | null
  engine_run_id: string | null
  trace_path: string | null
  stop_requested: boolean
  plan: PlanPayload | null
  last_seq?: number
  /** services 重启后 jobs.json 里没有的 run：详情由 trace 回放，只读 */
  readonly?: boolean
  source?: 'job' | 'trace'
}

export interface RunList {
  items: RunDetail[]
  count: number
  queue: number
}

export const TERMINAL_STATUS: RunStatus[] = ['succeeded', 'failed', 'stopped', 'interrupted']

export function isTerminal(status: RunStatus | undefined): boolean {
  return status !== undefined && TERMINAL_STATUS.includes(status)
}

export function getRun(runId: string): Promise<RunDetail> {
  return request<RunDetail>(`/api/runs/${encodeURIComponent(runId)}`)
}

export function listRuns(): Promise<RunList> {
  return request<RunList>('/api/runs')
}

export interface CreateRunInput {
  goal: string
  /** 留空 = 用配置里的 engine.max_turns。 */
  max_turns?: number
  /** false = 计划卡照发但不等确认，直接开始。 */
  plan_gate?: boolean
}

/** POST /api/runs：入队一个 run，返回 202 + 新 run 的初始状态。 */
export function createRun(input: CreateRunInput): Promise<{ run: RunDetail }> {
  return request<{ run: RunDetail }>('/api/runs', {
    method: 'POST',
    body: JSON.stringify(input),
  })
}

export function stopRun(runId: string): Promise<{ ok: boolean; run: RunDetail }> {
  return request(`/api/runs/${encodeURIComponent(runId)}/stop`, { method: 'POST' })
}

export function injectRun(
  runId: string,
  text: string
): Promise<{ ok: boolean; run: RunDetail }> {
  return request(`/api/runs/${encodeURIComponent(runId)}/inject`, {
    method: 'POST',
    body: JSON.stringify({ text }),
  })
}

export function decidePlan(
  runId: string,
  action: 'start' | 'revise' | 'cancel',
  text = ''
): Promise<{ ok: boolean; run: RunDetail }> {
  return request(`/api/runs/${encodeURIComponent(runId)}/plan`, {
    method: 'POST',
    body: JSON.stringify({ action, text }),
  })
}

export function resumeRun(runId: string): Promise<{ run: RunDetail }> {
  return request(`/api/runs/${encodeURIComponent(runId)}/resume`, { method: 'POST' })
}

// ---------------------------------------------------------------- 看板（S4）

export interface HealthItem {
  label: string
  status: 'ok' | 'warn'
  detail: string
}

export interface HealthQueue extends HealthItem {
  length: number
  running: number
}

export interface MetricsKpis {
  runs: number
  pass_rate: number
  passed: number
  failed: number
  unverified: number
  duration_p50_s: number | null
  duration_p90_s: number | null
  duration_n: number
  tokens_total: number
  prompt_tokens: number
  tokens_cached: number
  cache_ratio: number
  tool_failures: number
  repairs: number
  turns: number
}

export interface DailyBucket {
  date: string
  runs: number
  passed: number
  failed: number
  tokens: number
  tool_failures: number
  by_model: Record<string, number>
}

export interface ModelUsage {
  model: string
  calls: number
  runs: number
  tokens: number
  cached_tokens: number
  prompt_tokens: number
  cache_ratio: number
}

export interface GapItem {
  label: string
  count: number
  runs: number
}

export interface MetricsSummary {
  window: { days: number; since: string; until: string; timezone: string }
  kpis: MetricsKpis
  daily: DailyBucket[]
  models: ModelUsage[]
  gaps: GapItem[]
  health: { kernel: HealthItem; arcgis: HealthItem; model: HealthItem; queue: HealthQueue }
  /** 全部 run 数（不受窗口限制） */
  total_runs: number
  runs_in_window: number
  /** 退化输入（空目录/全坏/样本过少）的说明；正常为 null */
  warning: string | null
}

export function getMetrics(days = 7): Promise<MetricsSummary> {
  return request<MetricsSummary>(`/api/metrics/summary?days=${days}`)
}

export interface TraceRow {
  run_id: string
  started_at: string
  finished_at: string | null
  day: string
  status: string
  turns: number
  tokens: number
  cached_tokens: number
  duration_s: number | null
  verdict: 'pass' | 'fail' | null
  passed: boolean | null
  goal: string
  problems: string[]
  repairs: number
  tool_failures: number
  models: Record<string, number>
  event_count: number
}

export type VerdictFilter = '' | 'pass' | 'fail' | 'none'

export interface TraceFilters {
  since?: string
  until?: string
  status?: string
  pass?: VerdictFilter
  q?: string
  limit?: number
  offset?: number
}

export interface TraceList {
  items: TraceRow[]
  /** 本页条数 */
  count: number
  /** 筛选后的总条数 */
  total: number
  limit: number
  offset: number
}

export function listTraces(filters: TraceFilters = {}): Promise<TraceList> {
  const params = new URLSearchParams()
  const push = (key: string, value: string | number | undefined) => {
    if (value !== undefined && value !== '') params.set(key, String(value))
  }
  push('since', filters.since)
  push('until', filters.until)
  push('status', filters.status)
  push('pass', filters.pass)
  push('q', filters.q)
  push('limit', filters.limit)
  push('offset', filters.offset)
  const query = params.toString()
  return request<TraceList>(`/api/traces${query ? `?${query}` : ''}`)
}

export interface TraceEvent {
  seq: number
  ts: string
  event: string
  payload?: Record<string, unknown>
}

export interface TraceDetail extends TraceRow {
  events: TraceEvent[]
  report: string | null
}

export function getTraceDetail(runId: string): Promise<TraceDetail> {
  return request<TraceDetail>(`/api/traces/${encodeURIComponent(runId)}`)
}

/** 下载地址：浏览器直接打开，token 走查询参数（下载链接带不了请求头）。 */
export function traceDownloadUrl(runId: string, kind: 'jsonl' | 'md'): string {
  const url = apiUrl(`/api/traces/${encodeURIComponent(runId)}/download`, { kind })
  const token = readToken()
  if (token) url.searchParams.set('token', token)
  return url.toString()
}

// ---------------------------------------------------------------- 配置（S5）

export interface ConfigActive {
  /** 供应商预设 id（`custom` = 反代/自建），只是元信息。 */
  provider: string
  model: string
  api_base: string
  /** 只给后 4 位；没有 key 或 key 太短时为空串。 */
  key_tail: string
  /** 配置文件不存在时为 false：前端据此走首次配置引导。 */
  config_exists: boolean
}

export interface ConfigMeta {
  /** 服务端当前工作区（绝对路径，本机控制台只读展示）。 */
  workspace: string
  path: string
  backup_dir: string
  size_bytes: number
  /** 配置文件不存在时为 false；此时 GET 返回 {} + sha256:""。 */
  config_exists: boolean
  /** 有排队/运行中的 run 时为 true：此时写配置一律 409。 */
  busy: boolean
  busy_run: string | null
  writable: boolean
  /** 「当前生效」摘要：真正被引擎读取的 model/api_base/key 尾部。 */
  active: ConfigActive
}

export interface EngineBudget {
  max_turns?: number
  max_repairs?: number
  context_soft_limit_tokens?: number
  kernel_idle_timeout_seconds?: number
  kernel_exec_timeout_seconds?: number
  [key: string]: unknown
}

export interface LlmConfig {
  provider?: string
  model?: string
  /** 空串 = 官方默认地址；反代通常以 `/v1` 结尾。 */
  api_base?: string
  /** 单次请求超时（秒），服务端限定 5 ~ 3600。 */
  timeout?: number
  /** 服务端永远返回脱敏值（`sk-…后4位`），不会回传真 key。 */
  api_key?: string
  fallback_models?: string[]
  routing_rules?: Record<string, string[]>
  engine?: EngineBudget
  temperature?: number
  max_tokens?: number
  [key: string]: unknown
}

/** GET /api/config：脱敏配置 + 乐观并发令牌 + 运行中标志。 */
export type ConfigResponse = LlmConfig & { sha256: string; _meta: ConfigMeta }

export interface ConfigDiffRow {
  path: string
  before: unknown
  after: unknown
  secret?: boolean
}

export interface ConfigSaveResult {
  ok: boolean
  sha256: string
  backup: string | null
  diff: ConfigDiffRow[]
  saved_at: string
  path: string
  restored_from?: string
}

export interface ConfigBackup {
  name: string
  size_bytes: number
  created_at: string
  sha256: string
}

export interface ConfigBackupList {
  items: ConfigBackup[]
  count: number
  dir: string
}

export type EngineNumberKey =
  | 'max_turns'
  | 'max_repairs'
  | 'context_soft_limit_tokens'
  | 'kernel_idle_timeout_seconds'
  | 'kernel_exec_timeout_seconds'

/** 只带改动字段的补丁；服务端按白名单再校验一次。 */
export interface ConfigPatch {
  provider?: string
  model?: string
  api_base?: string
  timeout?: number
  routing_rules?: Record<string, string[]>
  fallback_models?: string[]
  engine?: Partial<Record<EngineNumberKey, number>>
  temperature?: number
  max_tokens?: number
}

export interface ProviderPreset {
  id: string
  name: string
  /** 空串表示预设不自带地址（`custom`）。 */
  default_base: string
  key_required: boolean
  hint: string
}

export interface ProviderList {
  items: ProviderPreset[]
  count: number
}

export interface ConfigTestRequest {
  provider?: string
  api_base: string
  model?: string
  /** 省略 = 后端读已保存的真 key；脱敏值会被拒绝。 */
  api_key?: string
}

export interface ConfigTestResult {
  ok: boolean
  latency_ms: number
  /** 最多 20 个候选模型名。 */
  models: string[]
  /** 失败原因（后端已脱敏）；成功时为空串。 */
  error: string
  provider?: string
  api_base?: string
  model?: string
  key_source?: 'config' | 'request' | 'none'
  has_key?: boolean
}

export function getConfig(): Promise<ConfigResponse> {
  return request<ConfigResponse>('/api/config')
}

export function getProviders(): Promise<ProviderList> {
  return request<ProviderList>('/api/config/providers')
}

/** 现场测连接：失败也返回 200 + `ok:false`，只有请求本身出错才抛。 */
export function testConnection(body: ConfigTestRequest): Promise<ConfigTestResult> {
  return request<ConfigTestResult>('/api/config/test', {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

export function saveConfig(
  patch: ConfigPatch,
  baseSha: string,
  apiKey?: string
): Promise<ConfigSaveResult> {
  const body: { base_sha256: string; patch: ConfigPatch; api_key?: string } = {
    base_sha256: baseSha,
    patch,
  }
  // 留空 = 不改：不要把空串/脱敏值发上去。
  if (apiKey && apiKey.trim()) body.api_key = apiKey.trim()
  return request<ConfigSaveResult>('/api/config', {
    method: 'PUT',
    body: JSON.stringify(body),
  })
}

export function listConfigBackups(): Promise<ConfigBackupList> {
  return request<ConfigBackupList>('/api/config/backups')
}

export function revertConfig(name: string): Promise<ConfigSaveResult> {
  return request<ConfigSaveResult>(`/api/config/revert/${encodeURIComponent(name)}`, {
    method: 'POST',
  })
}

// ---------------------------------------------------------------- 数据资产（S6）

export interface DatasetItem {
  path: string
  kind: string
  container: string
  name: string
  size: number
  mtime: number
  crs_name: string
  wkid: number | null
  geom_type: string
  feature_count: number | null
  /** [minX, minY, maxX, maxY]；catalog 只存 bbox，没有几何。 */
  extent: number[] | null
  field_count: number
  source: string
  license: string
  acquired_at: string
}

export interface DatasetList {
  items: DatasetItem[]
  count: number
  limit: number
  offset: number
  has_more: boolean
  counts: Record<string, number>
}

export interface DatasetField {
  name: string
  alias: string
  type: string
  length: number
  nullable: boolean
  samples: unknown[]
  null_ratio: number | null
  numeric: boolean
  sample_count: number
  min?: number
  max?: number
  range_basis?: string
}

export interface DatasetProfile extends DatasetItem {
  fields: DatasetField[]
  numeric_fields: string[]
  raster: Record<string, unknown>
  extra: Record<string, unknown>
}

export interface DatasetQuery {
  kind?: string
  q?: string
  limit?: number
  offset?: number
}

export function listDatasets(query: DatasetQuery = {}): Promise<DatasetList> {
  const params = new URLSearchParams()
  if (query.kind) params.set('kind', query.kind)
  if (query.q) params.set('q', query.q)
  if (query.limit !== undefined) params.set('limit', String(query.limit))
  if (query.offset !== undefined) params.set('offset', String(query.offset))
  const search = params.toString()
  return request<DatasetList>(`/api/catalog/datasets${search ? `?${search}` : ''}`)
}

/** 路径走 path 段（可能含空格/中文/反斜杠），先整段编码再拼。 */
export function getDatasetProfile(path: string): Promise<DatasetProfile> {
  return request<DatasetProfile>(
    `/api/catalog/datasets/${encodeURIComponent(path)}/profile`
  )
}

export interface PreviewCacheMeta {
  hit: boolean
  key: string
  elapsed_ms: number
  /** 只在未命中时出现：真正读几何花了多久。 */
  read_ms?: number
}

export interface GeoJSONGeometry {
  type: string
  coordinates?: unknown
}

export interface GeoJSONFeature {
  type: 'Feature'
  properties: Record<string, unknown>
  geometry: GeoJSONGeometry | null
}

export interface LayerPreview {
  path: string
  name: string
  kind: string
  wkid: number | null
  crs_name: string
  /** 非 4326/3857 时给「未做投影变换」，页面必须显式横幅。 */
  crs_warning: string | null
  bbox: [number, number, number, number]
  feature_count: number | null
  field_count: number
  returned_features: number
  point_count: number
  max_features: number
  truncated: boolean
  oversized?: boolean
  cache: PreviewCacheMeta
  geojson: { type: 'FeatureCollection'; features: GeoJSONFeature[] }
}

export function getLayerPreview(path: string, maxFeatures = 400): Promise<LayerPreview> {
  const params = new URLSearchParams({ path, max_features: String(maxFeatures) })
  return request<LayerPreview>(`/api/catalog/preview?${params.toString()}`)
}

// ---------------------------------------------------------------- 成果包（S6）

export interface ReportItem {
  name: string
  dir: string
  kind: 'package' | 'provenance'
  size: number
  mtime: number
  url: string
}

export interface ReportList {
  items: ReportItem[]
  count: number
  root: string
  names: string[]
}

export function listReports(): Promise<ReportList> {
  return request<ReportList>('/api/reports')
}

/** iframe / 下载链接带不了请求头，token 走查询参数（服务端两种都认）。 */
export function reportUrl(name: string, download = false): string {
  const encoded = name
    .split('/')
    .map((segment) => encodeURIComponent(segment))
    .join('/')
  const url = apiUrl(`/api/reports/${encoded}`, download ? { download: 'true' } : {})
  const token = readToken()
  if (token) url.searchParams.set('token', token)
  return url.toString()
}
