import { createRootRoute, createRoute, createRouter, Link } from '@tanstack/react-router'

import { AppShell } from '@/components/layout/app-shell'
import { PageHeader } from '@/components/layout/page-header'
import { Button } from '@/components/ui/button'
import { Empty } from '@/components/ui/empty'
import { ConfigPage } from '@/routes/config'
import { DataAssetsPage } from '@/routes/data'
import { EnvCheckPage } from '@/routes/env-check'
import { NewRunPage } from '@/routes/new-run'
import { OverviewPage } from '@/routes/overview'
import { PackagesPage } from '@/routes/packages'
import { RunDetailPage } from '@/routes/run-detail'
import { RunsPage } from '@/routes/runs'

function NotFoundPage() {
  return (
    <>
      <PageHeader title="页面不存在" description="链接可能已过期，或这一段还没建。" />
      <Empty
        title="404"
        description="回到概览继续操作。"
        action={
          <Button asChild size="sm">
            <Link to="/">返回概览</Link>
          </Button>
        }
      />
    </>
  )
}

const rootRoute = createRootRoute({ component: AppShell })

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  component: OverviewPage,
})

const runsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/runs',
  component: RunsPage,
})

const newRunRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/runs/new',
  component: NewRunPage,
})

const runDetailRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/runs/$runId',
  component: RunDetailPage,
})

const dataRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/data',
  component: DataAssetsPage,
})

const packagesRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/packages',
  component: PackagesPage,
})

// 成果包页在 S2 的导航里叫 /packages；S6 单里写的路径是 /reports。两个都留，
// 免得书签/文档互相 404。
const reportsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/reports',
  component: PackagesPage,
})

const configRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/config',
  component: ConfigPage,
})

const envRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/env',
  component: EnvCheckPage,
})

const notFoundRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '$',
  component: NotFoundPage,
})

const routeTree = rootRoute.addChildren([
  indexRoute,
  runsRoute,
  newRunRoute,
  runDetailRoute,
  dataRoute,
  packagesRoute,
  reportsRoute,
  configRoute,
  envRoute,
  notFoundRoute,
])

export const router = createRouter({
  routeTree,
  defaultPreload: 'intent',
  scrollRestoration: true,
})

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}
