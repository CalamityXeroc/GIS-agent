import { PageHeader } from '@/components/layout/page-header'
import { NewRunForm } from '@/components/runs/new-run-form'

export function NewRunPage() {
  return (
    <>
      <PageHeader
        title="新建运行"
        description="提交一个任务并进入单 worker 队列排队执行。"
      />
      <NewRunForm />
    </>
  )
}
