import { Link, useRouterState } from '@tanstack/react-router'
import {
  Database,
  History,
  LayoutDashboard,
  Package,
  PlayCircle,
  Settings2,
  Stethoscope,
  type LucideIcon,
} from 'lucide-react'
import type * as React from 'react'

import { cn } from '@/lib/utils'

/** 三段式导航（§3）：只放本项目真实存在的七块内容。 */
export type NavPath =
  | '/'
  | '/runs'
  | '/runs/new'
  | '/data'
  | '/packages'
  | '/config'
  | '/env'

export interface NavItem {
  to: NavPath
  label: string
  icon: LucideIcon
}

export interface NavGroupData {
  label: string
  items: NavItem[]
}

export const NAV_GROUPS: NavGroupData[] = [
  {
    label: '运行',
    items: [
      { to: '/', label: '概览', icon: LayoutDashboard },
      { to: '/runs', label: '运行历史', icon: History },
      { to: '/runs/new', label: '新建运行', icon: PlayCircle },
    ],
  },
  {
    label: '数据',
    items: [
      { to: '/data', label: '数据资产', icon: Database },
      { to: '/packages', label: '成果包', icon: Package },
    ],
  },
  {
    label: '系统',
    items: [
      { to: '/config', label: '配置模型', icon: Settings2 },
      { to: '/env', label: '环境自检', icon: Stethoscope },
    ],
  },
]

const ALL_PATHS = NAV_GROUPS.flatMap((group) => group.items.map((item) => item.to))

/** 取「最长匹配」：/runs/new 命中时不会再同时点亮 /runs。 */
export function pickActivePath(pathname: string): NavPath | undefined {
  return ALL_PATHS.filter((to) =>
    to === '/' ? pathname === '/' : pathname === to || pathname.startsWith(`${to}/`)
  ).sort((a, b) => b.length - a.length)[0]
}

function NavLinkItem({
  item,
  active,
  onNavigate,
}: {
  item: NavItem
  active: boolean
  onNavigate?: () => void
}) {
  const Icon = item.icon
  return (
    <Link
      to={item.to}
      onClick={onNavigate}
      // exact 只是关掉路由默认的前缀匹配（/runs 会把 /runs/new 也算 active，
      // 出现两个 aria-current）；高亮与否统一由 pickActivePath 的最长匹配决定。
      activeOptions={{ exact: true }}
      aria-current={active ? 'page' : undefined}
      className={cn(
        'group flex h-9 items-center gap-2.5 rounded-md px-2.5 text-sm transition-colors',
        active
          ? 'bg-sidebar-accent font-medium text-sidebar-accent-foreground'
          : 'text-foreground/75 hover:bg-accent/60 hover:text-foreground'
      )}
    >
      <Icon
        className={cn(
          'size-4 shrink-0',
          active ? 'text-primary' : 'text-muted-foreground group-hover:text-foreground'
        )}
      />
      <span className="truncate">{item.label}</span>
    </Link>
  )
}

export interface AppSidebarProps extends React.ComponentProps<'nav'> {
  /** 移动端抽屉里不吸顶、不固定宽度。 */
  mobile?: boolean
  onNavigate?: () => void
}

export function AppSidebar({
  mobile = false,
  onNavigate,
  className,
  ...props
}: AppSidebarProps) {
  const pathname = useRouterState({ select: (state) => state.location.pathname })
  const activePath = pickActivePath(pathname)

  return (
    <nav
      data-slot="app-sidebar"
      aria-label="主导航"
      className={cn(
        'flex flex-col gap-4 bg-sidebar px-3 py-4',
        mobile
          ? 'h-full w-full'
          : 'sticky top-[var(--app-header-height)] hidden h-[calc(100dvh-var(--app-header-height))] w-[var(--app-sidebar-width)] shrink-0 overflow-y-auto border-r border-sidebar-border md:flex',
        className
      )}
      {...props}
    >
      {NAV_GROUPS.map((group) => (
        <div key={group.label} className="flex flex-col gap-1">
          <div className="px-2.5 pb-1 text-xs font-medium text-muted-foreground">
            {group.label}
          </div>
          {group.items.map((item) => (
            <NavLinkItem
              key={item.to}
              item={item}
              active={activePath === item.to}
              onNavigate={onNavigate}
            />
          ))}
        </div>
      ))}

      <div className="mt-auto flex items-center gap-1.5 rounded-md border border-sidebar-border px-2.5 py-1.5 text-[11px] text-muted-foreground">
        <span className="size-1.5 shrink-0 rounded-full bg-success" aria-hidden="true" />
        只读控制台 · 本机
      </div>
    </nav>
  )
}
