import { Outlet, useRouterState } from '@tanstack/react-router'
import { useEffect, useState } from 'react'

import { AppSidebar } from '@/components/layout/app-sidebar'
import { Header } from '@/components/layout/header'
import { NavSheet } from '@/components/layout/nav-sheet'
import { Toaster } from '@/components/ui/sonner'
import { syncThemeClass } from '@/lib/theme'

/** 骨架：48px 顶栏 + 208px 三组侧栏 + 内容区（§2.1）。 */
export function AppShell() {
  const [mobileNavOpen, setMobileNavOpen] = useState(false)
  const pathname = useRouterState({ select: (state) => state.location.pathname })

  // 数据资产是沉浸式工作台（DAIR-345）：目录树要贴 x=0、地图要占满横向，
  // 所以这一页不渲染顶栏和导航侧栏，导航改由工作台自己的命令栏（☰）打开同一个抽屉。
  // 带尾斜杠的 /data/ 也要算进来（DAIR-361）：TanStack Router 不做归一，手输地址
  // 或旧书签会停在 /data/，与 pickActivePath 的「/data/ 属于 /data」约定保持一致。
  const immersive = pathname === '/data' || pathname.startsWith('/data/')

  useEffect(() => {
    syncThemeClass()
  }, [])

  return (
    <div className="min-h-dvh bg-background text-foreground">
      {immersive ? null : <Header onOpenMenu={() => setMobileNavOpen(true)} />}

      <div className={immersive ? 'flex' : 'flex min-h-[calc(100dvh-var(--app-header-height))]'}>
        {immersive ? null : <AppSidebar />}
        <main
          className={
            immersive ? 'min-w-0 flex-1' : 'min-w-0 flex-1 px-4 py-5 sm:px-6 lg:px-8'
          }
        >
          <Outlet />
        </main>
      </div>

      <NavSheet open={mobileNavOpen} onOpenChange={setMobileNavOpen} />

      <Toaster />
    </div>
  )
}
