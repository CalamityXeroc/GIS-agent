import { Sheet, SheetContent, SheetTitle } from '@/components/ui/sheet'

import { AppSidebar } from './app-sidebar'

/**
 * 导航抽屉（DAIR-345）：小屏顶栏的 ☰ 与沉浸式工作台（/data）共用这一个抽屉。
 * 移动布局下不吸顶、不固定宽度，点任一导航项后自动收起。
 */
export function NavSheet({
  open,
  onOpenChange,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="left" className="gap-0 p-0">
        <SheetTitle className="sr-only">导航</SheetTitle>
        <AppSidebar mobile onNavigate={() => onOpenChange(false)} />
      </SheetContent>
    </Sheet>
  )
}
