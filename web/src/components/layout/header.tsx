import { Link } from '@tanstack/react-router'
import { Menu, Moon, Search, Sun } from 'lucide-react'

import { useThemeStore } from '@/lib/theme'

export interface HeaderProps {
  onOpenMenu: () => void
}

/** 顶栏：48px sticky，logo + 版本 + 搜索占位 + 主题切换 + 头像位（§2.1）。 */
export function Header({ onOpenMenu }: HeaderProps) {
  const theme = useThemeStore((state) => state.theme)
  const toggleTheme = useThemeStore((state) => state.toggle)

  return (
    <header className="sticky top-0 z-40 flex h-[var(--app-header-height)] items-center gap-3 border-b border-border bg-background/85 px-3 backdrop-blur-md sm:px-4">
      <button
        type="button"
        className="grid size-8 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground md:hidden"
        aria-label="打开导航"
        onClick={onOpenMenu}
      >
        <Menu className="size-4" />
      </button>

      <Link to="/" className="flex shrink-0 items-center gap-2">
        <span
          className="grid size-6 place-items-center rounded-md bg-primary text-primary-foreground"
          aria-hidden="true"
        >
          <svg viewBox="0 0 24 24" fill="none" className="size-3.5">
            <path
              d="M3 7.5 12 3l9 4.5-9 4.5-9-4.5Z"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinejoin="round"
            />
            <path
              d="M3 12.5 12 17l9-4.5"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinejoin="round"
            />
          </svg>
        </span>
        <span className="text-[15px] font-semibold tracking-tight text-foreground">
          GIS Agent
        </span>
        <span className="rounded-full border border-border px-1.5 py-px text-[10px] font-medium tabular-nums text-muted-foreground">
          v{__APP_VERSION__}
        </span>
      </Link>

      {/* 搜索位：S2 只占位，不接检索。 */}
      <div className="mx-auto hidden w-full max-w-md justify-center lg:flex">
        <div
          role="search"
          aria-label="搜索（占位，尚未实现）"
          className="flex h-8 w-full items-center gap-2 rounded-full border border-border bg-card px-3 text-xs text-muted-foreground"
        >
          <Search className="size-3.5 shrink-0" />
          <span className="truncate">搜索运行、数据资产…</span>
          <kbd className="ml-auto hidden shrink-0 rounded-sm border border-border px-1 py-px text-[10px] sm:block">
            ⌘K
          </kbd>
        </div>
      </div>

      <div className="ml-auto flex items-center gap-1">
        <button
          type="button"
          onClick={toggleTheme}
          aria-label={theme === 'dark' ? '切换到浅色模式' : '切换到深色模式'}
          title={theme === 'dark' ? '切换到浅色模式' : '切换到深色模式'}
          className="grid size-8 place-items-center rounded-full text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground"
        >
          {theme === 'dark' ? <Sun className="size-4" /> : <Moon className="size-4" />}
        </button>
        <div className="mx-1 h-4 w-px bg-border" aria-hidden="true" />
        <div
          className="grid size-7 place-items-center rounded-full border border-border bg-muted text-[10px] font-semibold tracking-wide text-muted-foreground"
          title="本机用户"
          aria-label="用户头像位"
        >
          GA
        </div>
      </div>
    </header>
  )
}
