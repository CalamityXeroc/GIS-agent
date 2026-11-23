import type { DatasetItem, DatasetProfile } from '@/lib/api'
import { cn } from '@/lib/utils'

import { formatNumber, kindLabel } from '../format'
import type { GisLayer } from '../store'
import { Field, Hint, Row, Section, Segmented } from './parts'

export interface GeneralTabProps {
  item: DatasetItem
  layer: GisLayer | null
  profile?: DatasetProfile
}

/** 「常规」tab：这一层在工程里的状态（不是数据本身，数据看元数据）。 */
export function GeneralTab({ item, layer, profile }: GeneralTabProps) {
  const isRaster = item.kind === 'raster'
  const geometry = isRaster ? 'Raster' : item.geom_type || profile?.geom_type || '—'
  const count = profile?.feature_count ?? item.feature_count

  return (
    <>
      <Section title="图层">
        <Field label="名称" value={item.name} />
        <Field label="类型" value={kindLabel(item.kind)} />
        <Field label="几何类型" value={geometry} />
        <Field label="要素数" value={isRaster ? '—（栅格没有要素）' : formatNumber(count)} />
        <Field
          label="可见"
          value={
            layer ? (
              layer.visible ? (
                '是'
              ) : (
                '否'
              )
            ) : (
              <span className="text-muted-foreground">未加进地图</span>
            )
          }
        />
        <Field
          label="不透明度"
          value={layer ? `${Math.round(layer.style.opacity * 100)}%` : '—'}
        />
        <Field label="最小比例尺" value="无（全比例尺显示）" />
        <Field label="最大比例尺" value="无" />
        <Field label="源路径" value={item.path} mono wrap />
      </Section>

      <Section title="显示">
        <Row label="名称标注">
          <Segmented
            options={[
              { value: 'off', label: '关', title: '本期不做：离线环境没有 glyphs 字体依赖' },
              { value: 'field', label: '标注字段', title: '本期不做（缺 glyphs）' },
            ]}
            value="off"
            onChange={() => undefined}
          />
        </Row>
        <Row label="缩放范围">
          <Segmented
            options={[
              { value: 'all', label: '全图' },
              { value: 'custom', label: '自定义', title: '本期留空：比例尺区间持久化需要工程文件' },
            ]}
            value="all"
            onChange={() => undefined}
          />
        </Row>
        <Hint>
          标注（label）本期不做：离线构建没有 glyphs 字体依赖，硬上会在图上画成方块。
        </Hint>
      </Section>

      {layer ? (
        <p className={cn('mt-1 text-[10.5px] leading-[1.55] text-muted-foreground')}>
          绘制顺序在左侧图层栈里拖 ⠿ 调整（栈顶最后绘制）；样式改动即时生效，但
          <b className="text-foreground/75">刷新页面即丢</b>——样式持久化接口还没做。
        </p>
      ) : null}
    </>
  )
}
