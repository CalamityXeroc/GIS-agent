import { Stethoscope } from 'lucide-react'

import { PageHeader } from '@/components/layout/page-header'
import { Empty } from '@/components/ui/empty'

export function EnvCheckPage() {
  return (
    <>
      <PageHeader title="环境自检" description="ArcPy 桥接、依赖与模板环境的健康检查。" />
      <Empty
        icon={Stethoscope}
        title="待实现"
        description="自检项列表与一键复检将在后续单里接入 envcheck 模块。"
      />
    </>
  )
}
