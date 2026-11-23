import { useQuery } from '@tanstack/react-query'
import { Link, useNavigate } from '@tanstack/react-router'
import {
  CircleAlert,
  CornerDownLeft,
  Cpu,
  FolderOpen,
  ListChecks,
  LoaderCircle,
  Play,
  Target,
  Terminal,
} from 'lucide-react'
import { useRef, useState } from 'react'
import { toast } from 'sonner'

import { Button } from '@/components/ui/button'
import { TitledCard } from '@/components/ui/titled-card'
import { ApiError, createRun, getConfig } from '@/lib/api'
import {
  helpText,
  matchSlashCommands,
  readAutoMode,
  resolveComposerInput,
  writeAutoMode,
  type SlashCommand,
} from '@/lib/slash-commands'
import { cn } from '@/lib/utils'

const INPUT_CLASS =
  'w-full rounded-md border border-border bg-background px-3 text-sm text-foreground outline-none placeholder:text-muted-foreground/70 focus:border-primary/50'

const HINT_CLASS = 'text-[11px] text-muted-foreground'

/** 留空合法（走配置默认）；只接受 ≥ 1 的整数。 */
function maxTurnsError(raw: string): string | null {
  const text = raw.trim()
  if (!text) return null
  if (!/^\d+$/.test(text)) return '只能填正整数；留空表示用配置默认'
  if (Number(text) < 1) return '至少 1 轮；留空表示用配置默认'
  return null
}

/** 纯展示开关：受控、可键盘操作（role=switch），不做动画之外的副作用。 */
function Switch({
  checked,
  onChange,
  label,
}: {
  checked: boolean
  onChange: (next: boolean) => void
  label: string
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={() => onChange(!checked)}
      className={cn(
        'relative h-5 w-9 shrink-0 rounded-full border transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring/40',
        checked ? 'border-primary bg-primary' : 'border-border bg-muted'
      )}
    >
      <span
        className={cn(
          'absolute top-0.5 size-3.5 rounded-full transition-all',
          checked ? 'left-[1.125rem] bg-primary-foreground' : 'left-0.5 bg-muted-foreground'
        )}
      />
    </button>
  )
}

/**
 * 新建运行表单：任务目标（必填）+ 工作区（只读）+ 最大轮次 + 计划卡确认。
 *
 * 任务目标输入框同时是控制台的命令行：``/`` 开头走斜杠指令（``/confirm on|off``
 * 开关免确认直执行、``/help`` 看指令表），不会当成任务目标提交。
 *
 * 提交成功后跳 ``/runs/{id}``；失败只就地显示服务端 ``detail``，不清空已填内容。
 */
export function NewRunForm() {
  const navigate = useNavigate()
  const [goal, setGoal] = useState('')
  const [maxTurns, setMaxTurns] = useState('')
  // 免确认直执行模式：存 localStorage，刷新后保持；默认关（仍弹计划卡）。
  const [autoMode, setAutoMode] = useState(() => readAutoMode())
  // 每次提交的计划卡确认开关：免确认模式关着时才显示，模式开着时整块隐藏。
  const [planGate, setPlanGate] = useState(true)
  const [helpOpen, setHelpOpen] = useState(false)
  const [menuHidden, setMenuHidden] = useState(false)
  const [slashIndex, setSlashIndex] = useState(0)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  // 按钮 disabled 之外再兜一层：Ctrl+Enter 连按时不能重复建 run。
  const inFlight = useRef(false)

  const config = useQuery({
    queryKey: ['config'],
    queryFn: getConfig,
    staleTime: 60_000,
    retry: false,
  })

  const turnsError = maxTurnsError(maxTurns)
  // 输入框内容 → 一个动作（提交 run / 切换模式 / 看帮助 / 报错）。
  const action = resolveComposerInput(goal, autoMode, planGate)
  const commandPending = action.kind === 'auto' || action.kind === 'help'
  const canSubmit =
    action.kind !== 'idle' && !submitting && (action.kind !== 'submit' || turnsError === null)
  const slashCandidates = menuHidden ? null : matchSlashCommands(goal)
  const activeSlash = Math.min(slashIndex, Math.max(0, (slashCandidates?.length ?? 1) - 1))

  const workspace = config.data?._meta.workspace
  const workspaceText = workspace
    ? workspace
    : config.isError
      ? '未知（配置接口不可用）'
      : '读取中…'
  // 只读展示：引擎下一个 run 会用的模型（改配置只影响之后的 run）。
  const active = config.data?._meta.active
  const modelText = active?.model || (config.isError ? '未知' : '未配置')

  /** 候选菜单里选中一条：带参数的补上名字+空格接着敲，不带参数的直接执行。 */
  function applySlashCandidate(command: SlashCommand) {
    setMenuHidden(true)
    setSlashIndex(0)
    if (command.args && command.args.length > 0) {
      setGoal(`/${command.name} `)
      return
    }
    setGoal('')
    void runComposer(`/${command.name}`)
  }

  /** 提交按钮 / Ctrl+Enter / 候选菜单共用：先看是不是指令，再建 run。 */
  async function runComposer(raw: string = goal) {
    if (inFlight.current) return
    const next = resolveComposerInput(raw, autoMode, planGate)
    if (next.kind === 'idle') return
    if (next.kind === 'error') {
      setError(next.message)
      return
    }
    if (next.kind === 'help') {
      setHelpOpen(true)
      setError('')
      return
    }
    if (next.kind === 'auto') {
      setAutoMode(next.enabled)
      writeAutoMode(next.enabled)
      setGoal('')
      setError('')
      setHelpOpen(false)
      toast.success(next.enabled ? '已开启免确认直执行' : '已关闭免确认直执行', {
        description: next.enabled
          ? '之后提交的任务不弹计划卡、提交即开跑（计划卡仍记进时间线）'
          : '之后提交的任务会先给计划卡，等你确认再执行',
      })
      return
    }

    const invalid = maxTurnsError(maxTurns)
    if (invalid) {
      setError(invalid)
      return
    }

    inFlight.current = true
    setSubmitting(true)
    setError('')
    try {
      const { run } = await createRun({
        goal: next.goal,
        plan_gate: next.planGate,
        ...(maxTurns.trim() ? { max_turns: Number(maxTurns.trim()) } : {}),
      })
      toast.success('已提交运行', {
        description: `run ${run.id} 已入队，正在打开实时详情…`,
      })
      await navigate({ to: '/runs/$runId', params: { runId: run.id } })
    } catch (exc) {
      // 409/400/422 的 detail 原样展示；表单内容保持不动，改完可直接重试。
      const message =
        exc instanceof ApiError
          ? `提交失败（${exc.status}）：${exc.message}`
          : `无法连接控制台服务：${exc instanceof Error ? exc.message : String(exc)}。确认 gis-agent web 仍在运行后重试。`
      setError(message)
    } finally {
      inFlight.current = false
      setSubmitting(false)
    }
  }

  return (
    <div className="mx-auto max-w-3xl space-y-4">
      <div className="flex items-start gap-2 rounded-lg border border-border bg-secondary/50 px-3.5 py-2.5 text-xs text-muted-foreground">
        <Terminal className="mt-px size-3.5 shrink-0 text-primary" />
        <p>
          本机控制台 · 队列串行执行：同一时刻只跑一个 run，提交后详情页会显示排队位置与实时时间线。
        </p>
      </div>

      <TitledCard
        title="任务目标"
        icon={Target}
        meta={<span className="tabular-nums">{goal.length} 字</span>}
      >
        <label htmlFor="run-goal" className="text-xs font-medium text-foreground">
          任务目标 <span className="text-destructive">*</span>
        </label>
        <p className={cn(HINT_CLASS, 'mt-1 mb-2')}>
          {autoMode
            ? '写清数据在哪、要产出什么、怎么算验收通过；免确认已开，提交即开跑，计划卡只记进时间线。'
            : '写清数据在哪、要产出什么、怎么算验收通过；引擎会先据此给一张计划卡。'}
        </p>
        <textarea
          id="run-goal"
          autoFocus
          rows={6}
          value={goal}
          aria-invalid={goal.trim() ? undefined : true}
          onChange={(event) => {
            setGoal(event.target.value)
            setMenuHidden(false)
            setSlashIndex(0)
            setError('')
          }}
          onKeyDown={(event) => {
            if (slashCandidates) {
              if (
                slashCandidates.length > 0 &&
                (event.key === 'ArrowDown' || event.key === 'ArrowUp')
              ) {
                event.preventDefault()
                const step = event.key === 'ArrowDown' ? 1 : -1
                setSlashIndex(
                  (index) => (index + step + slashCandidates.length) % slashCandidates.length
                )
                return
              }
              if (event.key === 'Escape') {
                event.preventDefault()
                setMenuHidden(true)
                return
              }
              // 候选区里按 Enter：有候选就应用，没候选（未知指令）直接报用法。
              if (event.key === 'Enter' && !event.metaKey && !event.ctrlKey && !event.shiftKey) {
                event.preventDefault()
                if (slashCandidates.length > 0) applySlashCandidate(slashCandidates[activeSlash])
                else void runComposer()
                return
              }
              if (event.key === 'Tab' && slashCandidates.length > 0) {
                event.preventDefault()
                applySlashCandidate(slashCandidates[activeSlash])
                return
              }
            }
            if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
              event.preventDefault()
              if (canSubmit) void runComposer()
            }
          }}
          placeholder="例：读 workspace/input/森林覆盖率.csv，按省汇总覆盖率并取 Top5，产出 output/森林覆盖率_top5.md（含表格与数据来源）"
          className={cn(INPUT_CLASS, 'resize-y py-2 leading-6')}
        />
        {slashCandidates ? (
          <ul
            role="listbox"
            aria-label="斜杠指令候选"
            className="mt-2 overflow-hidden rounded-md border border-border bg-card text-xs"
          >
            {slashCandidates.length > 0 ? (
              slashCandidates.map((command, index) => (
                <li key={command.name} role="option" aria-selected={index === activeSlash}>
                  <button
                    type="button"
                    onMouseDown={(event) => {
                      // 别让 textarea 先失焦：失焦后 Tab / ↑↓ 就不在输入框里了。
                      event.preventDefault()
                      applySlashCandidate(command)
                    }}
                    className={cn(
                      'flex w-full flex-wrap items-baseline gap-x-2 px-3 py-2 text-left',
                      index === activeSlash
                        ? 'bg-secondary text-foreground'
                        : 'text-muted-foreground'
                    )}
                  >
                    <span className="font-mono text-foreground">{command.usage}</span>
                    <span>{command.summary}</span>
                  </button>
                </li>
              ))
            ) : (
              <li className="px-3 py-2 text-muted-foreground">没有匹配的指令；输入 /help 看全部</li>
            )}
          </ul>
        ) : null}
        {helpOpen ? (
          <div className="mt-2 rounded-md border border-border bg-muted/40 px-3 py-2">
            <div className="flex items-center justify-between gap-2">
              <span className="text-[11px] font-medium text-foreground">斜杠指令</span>
              <button
                type="button"
                onClick={() => setHelpOpen(false)}
                className="text-[11px] text-muted-foreground hover:text-foreground"
              >
                关闭
              </button>
            </div>
            <pre className="mt-1 font-sans text-[11px] leading-5 whitespace-pre-wrap text-muted-foreground">
              {helpText(autoMode)}
            </pre>
          </div>
        ) : null}
        <div className="mt-2 flex items-center gap-1.5 text-[11px] text-muted-foreground">
          <CornerDownLeft className="size-3" />
          <span>Ctrl / ⌘ + Enter 提交</span>
        </div>
      </TitledCard>

      <TitledCard title="运行选项" icon={ListChecks} meta={autoMode ? '免确认直执行' : '可选'}>
        <div className="divide-y divide-border">
          <div className="flex flex-wrap items-start justify-between gap-3 pb-3">
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-1.5 text-xs font-medium text-foreground">
                <FolderOpen className="size-3.5 text-muted-foreground" />
                工作区
              </div>
              <p className={cn(HINT_CLASS, 'mt-1')}>服务端当前工作区，本机自用不做切换。</p>
            </div>
            <div
              title={workspace ?? workspaceText}
              className="max-w-[22rem] min-w-0 truncate rounded-md border border-border bg-muted/60 px-2.5 py-1.5 font-mono text-[11px] text-muted-foreground"
            >
              {workspaceText}
            </div>
          </div>

          <div className="flex flex-wrap items-start justify-between gap-3 py-3">
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-1.5 text-xs font-medium text-foreground">
                <Cpu className="size-3.5 text-muted-foreground" />
                当前模型
              </div>
              <p className={cn(HINT_CLASS, 'mt-1')}>
                来自 config/llm_config.json 的 model；要换模型去「配置模型 · 模型与 API」。
              </p>
            </div>
            <div
              title={active?.api_base || undefined}
              className="max-w-[22rem] min-w-0 truncate rounded-md border border-border bg-muted/60 px-2.5 py-1.5 font-mono text-[11px] text-muted-foreground"
            >
              {modelText}
            </div>
          </div>

          <div className="flex flex-wrap items-start justify-between gap-3 py-3">
            <div className="min-w-0 flex-1">
              <label htmlFor="run-max-turns" className="text-xs font-medium text-foreground">
                最大轮次
              </label>
              <p className={cn(HINT_CLASS, 'mt-1')}>
                一个 run 最多执行多少轮工具调用；留空 = 用配置默认（engine.max_turns）。
              </p>
            </div>
            <div className="w-28 shrink-0">
              <input
                id="run-max-turns"
                inputMode="numeric"
                value={maxTurns}
                placeholder="默认"
                aria-invalid={turnsError ? true : undefined}
                aria-describedby={turnsError ? 'run-max-turns-error' : undefined}
                onChange={(event) => setMaxTurns(event.target.value)}
                className={cn(
                  INPUT_CLASS,
                  'h-9 text-right tabular-nums',
                  turnsError && 'border-destructive focus:border-destructive'
                )}
              />
              {turnsError ? (
                <p id="run-max-turns-error" className="mt-1 text-[11px] text-destructive">
                  {turnsError}
                </p>
              ) : null}
            </div>
          </div>

          {autoMode ? null : (
            <div className="flex flex-wrap items-start justify-between gap-3 pt-3">
              <div className="min-w-0 flex-1">
                <div className="text-xs font-medium text-foreground">计划卡确认</div>
                <p className={cn(HINT_CLASS, 'mt-1')}>
                  {planGate
                    ? '开跑前先给一张计划卡，等你确认后再执行。'
                    : '直接开始，不再等确认（计划卡仍会记进时间线）。'}
                </p>
              </div>
              <Switch checked={planGate} onChange={setPlanGate} label="计划卡确认" />
            </div>
          )}
        </div>
      </TitledCard>

      {error ? (
        <div
          role="alert"
          className="flex items-start gap-2 rounded-lg border border-destructive/40 bg-destructive/10 px-3.5 py-2.5 text-xs text-destructive"
        >
          <CircleAlert className="mt-px size-3.5 shrink-0" />
          <p className="min-w-0 break-words">{error}</p>
        </div>
      ) : null}

      <div className="flex flex-wrap items-center gap-3">
        <Button type="button" onClick={() => void runComposer()} disabled={!canSubmit}>
          {submitting ? (
            <LoaderCircle className="size-4 animate-spin" />
          ) : (
            <Play className="size-4" />
          )}
          {commandPending ? '执行指令' : submitting ? '提交中…' : '提交并进入详情'}
        </Button>
        <span className={HINT_CLASS}>
          {turnsError
            ? '先修正最大轮次'
            : autoMode
              ? '免确认直执行已开：提交即开跑；/confirm off 恢复计划卡'
              : '输入 / 打开指令；提交后在详情页可停止 / 注入消息 / 批准计划'}
        </span>
        <Button variant="ghost" className="ml-auto" asChild>
          <Link to="/runs">取消</Link>
        </Button>
      </div>
    </div>
  )
}
