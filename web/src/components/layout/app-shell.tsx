import { Outlet } from '@tanstack/react-router'
import { useEffect, useState } from 'react'

import { AppSidebar } from '@/components/layout/app-sidebar'
import { Header } from '@/components/layout/header'
import { Toaster } from '@/components/ui/sonner'
import { Sheet, SheetContent, SheetTitle } from '@/components/ui/sheet'
import { syncThemeClass } from '@/lib/theme'

/** 骨架：48px 顶栏 + 208px 三组侧栏 + 内容区（§2.1）。 */
export function AppShell() {
  const [mobileNavOpen, setMobileNavOpen] = useState(false)

  useEffect(() => {
    syncThemeClass()
  }, [])

  return (
    <div className="min-h-dvh bg-background text-foreground">
      <Header onOpenMenu={() => setMobileNavOpen(true)} />

      <div className="flex min-h-[calc(100dvh-var(--app-header-height))]">
        <AppSidebar />
        <main className="min-w-0 flex-1 px-4 py-5 sm:px-6 lg:px-8">
          <Outlet />
        </main>
      </div>

      <Sheet open={mobileNavOpen} onOpenChange={setMobileNavOpen}>
        <SheetContent side="left" className="gap-0 p-0">
          <SheetTitle className="sr-only">导航</SheetTitle>
          <AppSidebar mobile onNavigate={() => setMobileNavOpen(false)} />
        </SheetContent>
      </Sheet>

      <Toaster />
    </div>
  )
}
