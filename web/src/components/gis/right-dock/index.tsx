import { Grid3x3, Info, Layers, X } from 'lucide-react'

import type { DatasetItem } from '@/lib/api'
import { cn } from '@/lib/utils'

import { crsBadge, kindLabel } from '../format'
import { useGisStore, type GisLayer, type RightTab } from '../store'
import { FieldsTab } from './fields-tab'
import { GeneralTab } from './general-tab'
import { MetadataTab } from './metadata-tab'
import { EmptyState } from './parts'
import { SymbologyTab } from './symbology-tab'
import { useDatasetProfile } from './use-profile'

export interface RightDockProps {
  selected: DatasetItem | null
  layer: GisLayer | null
  inLayerStack: boolean
  /** 预览首要素的几何类型：catalog 的 geom_type 对 geojson 是空的。 */
  geometryType: string | null
  width: number
  onClose: () => void
}

interface TabDef {
  key: RightTab
  label: string
  badge?: string
  /** 不可用的原因；给了就置灰并写进 title。 */
  disabledReason?: string
}

/**
 * 右 dock（312px）：常规 / 符号化⑤ / 字段 / 元数据④。
 *
 * 选中项存在 store（`rightTab`），所以切图层、切 tab、重开面板都不会丢状态；
 * 数据全部来自既有只读接口（profile / distinct / preview），缺的字段一律
 * 「待补 + 原因」，不留空表冒充。
 */
export function RightDock({
  selected,
  layer,
  inLayerStack,
  geometryType,
  width,
  onClose,
}: RightDockProps) {
  const rightTab = useGisStore((state) => state.rightTab)
  const setRightTab = useGisStore((state) => state.setRightTab)
  const profile = useDatasetProfile(selected?.path ?? null)

  const isRaster = selected?.kind === 'raster'
  const tabs: TabDef[] = [
    { key: 'general', label: '常规' },
    {
      key: 'symbology',
      label: '符号化',
      badge: '⑤',
      disabledReason: layer ? undefined : '该数据集还没加进地图',
    },
    { key: 'fields', label: '字段', disabledReason: isRaster ? '栅格没有属性字段' : undefined },
    { key: 'metadata', label: '元数据', badge: '④' },
  ]
  const active = tabs.find((tab) => tab.key === rightTab) ?? tabs[0]

  return (
    <aside
      className="flex shrink-0 flex-col overflow-hidden border-l border-border bg-card"
      style={{ width }}
      aria-label="图层属性"
      data-testid="right-dock"
    >
      <div className="flex h-[var(--gis-panel-head)] shrink-0 items-center gap-2 border-b border-border px-2.5">
        <span className="grid size-6 shrink-0 place-items-center rounded-[var(--gis-radius)] bg-accent text-primary">
          {isRaster ? <Grid3x3 className="size-3.5" /> : <Layers className="size-3.5" />}
        </span>
        <div className="min-w-0 flex-1">
          <b className="block truncate text-[12px] font-semibold">{selected?.name ?? '图层属性'}</b>
          <small className="block truncate text-[10.5px] text-muted-foreground">
            {selected
              ? `${kindLabel(selected.kind)}${crsBadge(selected) ? ` · ${crsBadge(selected)}` : ''}${
                  inLayerStack ? '' : ' · 未加入地图'
                }`
              : '未选择图层'}
          </small>
        </div>
        <button
          type="button"
          onClick={onClose}
          title="关闭右面板"
          aria-label="关闭右面板"
          className="grid size-5 shrink-0 place-items-center rounded-[3px] text-muted-foreground hover:bg-accent hover:text-foreground"
        >
          <X className="size-3.5" />
        </button>
      </div>

      <div
        role="tablist"
        aria-label="图层属性分区"
        className="flex h-[30px] shrink-0 items-stretch border-b border-border bg-background"
      >
        {tabs.map((tab) => {
          const disabled = Boolean(tab.disabledReason) || !selected
          return (
            <button
              key={tab.key}
              type="button"
              role="tab"
              aria-selected={active.key === tab.key}
              disabled={disabled}
              title={tab.disabledReason ?? tab.label}
              onClick={() => setRightTab(tab.key)}
              className={cn(
                'relative flex flex-1 items-center justify-center gap-0.5 text-[11.5px] disabled:cursor-not-allowed disabled:opacity-45',
                active.key === tab.key
                  ? 'bg-card font-semibold text-foreground after:absolute after:inset-x-2 after:bottom-[-1px] after:h-0.5 after:rounded after:bg-primary'
                  : 'text-muted-foreground hover:text-foreground'
              )}
            >
              {tab.label}
              {tab.badge ? (
                <span className="grid size-3 place-items-center rounded-full bg-primary text-[8px] leading-none font-bold text-primary-foreground">
                  {tab.badge}
                </span>
              ) : null}
            </button>
          )
        })}
      </div>

      <div className="min-h-0 flex-1 overflow-auto px-2.5 py-2" role="tabpanel">
        {!selected ? (
          <EmptyState title="未选择图层">
            <span>在左栏目录或图层栈里选一项，这里显示它的样式、字段与元数据。</span>
            <Info className="mt-1 size-3 opacity-40" />
          </EmptyState>
        ) : active.key === 'general' ? (
          <GeneralTab item={selected} layer={layer} profile={profile.data} />
        ) : active.key === 'symbology' ? (
          <SymbologyTab
            item={selected}
            layer={layer}
            profile={profile.data}
            geometryType={geometryType}
          />
        ) : active.key === 'fields' ? (
          <FieldsTab
            profile={profile.data}
            loading={profile.isLoading}
            error={(profile.error as Error | null) ?? null}
            isRaster={isRaster}
          />
        ) : (
          <MetadataTab
            item={selected}
            profile={profile.data}
            loading={profile.isLoading}
            error={(profile.error as Error | null) ?? null}
          />
        )}
      </div>
    </aside>
  )
}
