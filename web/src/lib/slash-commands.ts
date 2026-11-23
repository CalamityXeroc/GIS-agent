/**
 * 控制台的斜杠指令：命令表 + 解析（唯一数据源，输入框候选与 ``/help`` 都读它）。
 *
 * 语义与 CLI 的 ``src/gis_cli/ui/commands.py`` 一致：``/name <参数>``，
 * 空白分隔、名字大小写不敏感。web 侧不做终端里那些会话级命令（``/theme``、
 * ``/model`` …），只保留提交任务时用得上的：``/confirm`` 与 ``/help``。
 *
 * 这里全是纯函数（不碰 React），所以能直接编成 ESM 在 node 里测。
 */

export interface SlashCommand {
  /** 不带前导斜杠（``"confirm"``），展示时再补——否则会出现 ``//confirm``。 */
  name: string
  summary: string
  usage: string
  /** 参数候选（菜单里提示用；空 = 不带参数）。 */
  args?: readonly string[]
}

/**: 全部斜杠指令（唯一的命令表）。 */
export const SLASH_COMMANDS: readonly SlashCommand[] = [
  {
    name: 'confirm',
    summary: '免确认直执行：开 = 不弹计划卡直接开跑（默认关）',
    usage: '/confirm <on|off>',
    args: ['on', 'off'],
  },
  { name: 'help', summary: '列出可用指令与当前模式', usage: '/help' },
]

/**: 免确认直执行的默认值：关（即默认仍弹计划卡、``plan_gate=true``）。 */
export const AUTO_MODE_DEFAULT = false

/**: 开关存在 localStorage：本机自用，刷新后保持。 */
export const AUTO_MODE_STORAGE_KEY = 'gis-agent-web-auto-mode'

/** 只用到读写两个方法：真实 ``Storage`` 与测试里的假存储都能传。 */
export interface ModeStorage {
  getItem(key: string): string | null
  setItem(key: string, value: string): void
}

function defaultStorage(): ModeStorage | null {
  if (typeof window === 'undefined') return null
  try {
    return window.localStorage
  } catch {
    // Safari 隐私模式连访问 localStorage 都会抛。
    return null
  }
}

/**
 * 读免确认开关。
 *
 * 没存过 / 值不认识（手改过 localStorage）/ 存储不可用 → 一律回默认关闭，
 * 不能因为一个坏值把「确认」这层安全网静默关掉。
 */
export function readAutoMode(storage: ModeStorage | null = defaultStorage()): boolean {
  if (!storage) return AUTO_MODE_DEFAULT
  try {
    const raw = storage.getItem(AUTO_MODE_STORAGE_KEY)
    if (raw === 'on') return true
    if (raw === 'off') return false
    return AUTO_MODE_DEFAULT
  } catch {
    return AUTO_MODE_DEFAULT
  }
}

/** 写免确认开关；存储不可用（隐私模式 / 配额满）时静默失败，本次会话仍生效。 */
export function writeAutoMode(
  enabled: boolean,
  storage: ModeStorage | null = defaultStorage()
): void {
  try {
    storage?.setItem(AUTO_MODE_STORAGE_KEY, enabled ? 'on' : 'off')
  } catch {
    /* 写不进去不影响当前页面里的状态 */
  }
}

export interface ParsedSlash {
  /** 小写命令名；只敲了 ``/`` 时为空串。 */
  name: string
  /** 第一个参数（小写）；没有则空串。 */
  arg: string
  /** 整段参数原文（保留大小写，给需要自由文本的指令用）。 */
  rest: string
}

/** 解析 ``/name arg``；不是斜杠开头的输入返回 ``null``。 */
export function parseSlashCommand(raw: string): ParsedSlash | null {
  const text = raw.trim()
  if (!text.startsWith('/')) return null
  const body = text.slice(1).trim()
  if (!body) return { name: '', arg: '', rest: '' }
  const parts = body.split(/\s+/)
  const [name, ...rest] = parts
  return { name: name.toLowerCase(), arg: (rest[0] ?? '').toLowerCase(), rest: rest.join(' ') }
}

/** 按名字找指令（大小写不敏感；``/help`` 与 ``help`` 都认）。 */
export function findCommand(name: string): SlashCommand | undefined {
  const key = name.trim().toLowerCase().replace(/^\//, '')
  if (!key) return undefined
  return SLASH_COMMANDS.find((command) => command.name === key)
}

/**
 * 输入框正在敲命令名时的候选（``/`` 或 ``/co``）。
 *
 * 敲了空白（``/confirm o``）就返回 ``null``：那时已经是参数阶段，菜单该收起，
 * 否则候选会挡住正文。
 */
export function matchSlashCommands(raw: string): SlashCommand[] | null {
  const text = raw.trimStart()
  if (!text.startsWith('/')) return null
  if (/\s/.test(text)) return null
  const query = text.slice(1).toLowerCase()
  return SLASH_COMMANDS.filter((command) => command.name.startsWith(query))
}

/** 提交按钮该做什么：由输入原文 + 当前模式算出来，组件只负责执行。 */
export type ComposerAction =
  | { kind: 'idle' }
  | { kind: 'submit'; goal: string; planGate: boolean }
  | { kind: 'auto'; enabled: boolean }
  | { kind: 'help' }
  | { kind: 'error'; message: string }

function usageList(): string {
  return SLASH_COMMANDS.map((command) => command.usage).join('、')
}

/**
 * 把输入框内容翻译成一个动作。
 *
 * 关键契约：斜杠指令**永远不会**变成一次 run（不会被当成任务目标提交）；
 * 免确认开着时提交的 run 带 ``planGate=false``（不看那个已经隐藏的每-run 开关），
 * 关着时按每-run 开关来（默认 ``true``，与改造前一致）。
 */
export function resolveComposerInput(
  raw: string,
  autoMode: boolean,
  planGate = true
): ComposerAction {
  const text = raw.trim()
  if (!text || text === '/') return { kind: 'idle' }
  if (!text.startsWith('/')) {
    return { kind: 'submit', goal: text, planGate: !autoMode && planGate }
  }

  const parsed = parseSlashCommand(text)
  if (!parsed || !parsed.name) return { kind: 'idle' }
  const command = findCommand(parsed.name)
  if (!command) {
    return { kind: 'error', message: `未知指令 /${parsed.name}；可用：${usageList()}` }
  }
  if (command.name === 'help') return { kind: 'help' }
  if (parsed.arg === 'on') return { kind: 'auto', enabled: true }
  if (parsed.arg === 'off') return { kind: 'auto', enabled: false }
  return {
    kind: 'error',
    message: `用法：${command.usage}；当前${autoMode ? '已开启' : '已关闭'}`,
  }
}

/** ``/help`` 正文：命令表 + 当前模式（命令段由上面的表生成，不另抄一份）。 */
export function helpText(autoMode: boolean): string {
  const lines = ['可用指令（输入 / 会列出候选，↑↓ 选择、Enter 应用、Esc 取消）：']
  for (const command of SLASH_COMMANDS) {
    lines.push(`  ${command.usage}  ${command.summary}`)
  }
  lines.push(
    '',
    `免确认直执行：当前${autoMode ? '已开启（不弹计划卡，提交即开跑；计划卡仍记进时间线）' : '已关闭（开跑前先给计划卡，等你确认）'}`
  )
  return lines.join('\n')
}
