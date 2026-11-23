import { Loader2 } from 'lucide-react'
import { useEffect } from 'react'

import type { DatasetItem, DatasetProfile, DistinctResult } from '@/lib/api'

import { formatNumber } from '../format'
import {
  useGisStore,
  type GisLayer,
  type RasterStretch,
  type RendererMode,
} from '../store'
import { describeApiError } from './errors'
import { categoryColor, fillCategoryColors, RAMPS, rampOf, sampleRamp } from './palette'
import {
  ColorSwatch,
  EmptyState,
  Hint,
  RampBar,
  RangeRow,
  Row,
  Section,
  SelectRow,
  Todo,
  Unavailable,
} from './parts'
import { useDistinct } from './use-distinct'

export interface SymbologyTabProps {
  item: DatasetItem
  layer: GisLayer | null
  profile?: DatasetProfile
  /** 预览首要素的几何类型（catalog 的 geom_type 对 geojson 是空的）。 */
  geometryType: string | null
}

/** 「符号化」tab（spec §4 ⑤）：改的是图层样式，地图侧 setPaintProperty 即时生效。 */
export function SymbologyTab({ item, layer, profile, geometryType }: SymbologyTabProps) {
  if (!layer) {
    return (
      <EmptyState title="该数据集还没加进地图">
        <span>先在左侧目录双击（或图层栈的 +）把它加到地图上，才能改符号。</span>
      </EmptyState>
    )
  }
  return item.kind === 'raster' ? (
    <RasterSymbology layer={layer} profile={profile} />
  ) : (
    <VectorSymbology layer={layer} profile={profile} geometryType={geometryType} />
  )
}

// ------------------------------------------------------------------ 矢量
function VectorSymbology({
  layer,
  profile,
  geometryType,
}: {
  layer: GisLayer
  profile?: DatasetProfile
  geometryType: string | null
}) {
  const setLayerStyle = useGisStore((state) => state.setLayerStyle)
  const resetLayerStyle = useGisStore((state) => state.resetLayerStyle)
  const style = layer.style
  const fields = profile?.fields ?? []
  const numericFields = profile?.numeric_fields ?? []
  const isPoint = Boolean(geometryType && /Point/i.test(geometryType))

  const unique = useDistinct(
    layer.path,
    style.mode === 'unique' ? style.uniqueField : null
  )
  const graduated = useDistinct(
    layer.path,
    style.mode === 'graduated' ? style.graduatedField : null,
    { classes: style.graduatedClasses }
  )

  // 唯一值：类目色落到 store，地图侧才能按类目 match 着色（不猜、不留空）。
  const uniqueKeys = (unique.data?.values ?? []).map((entry) => String(entry.value))
  const uniqueSignature = uniqueKeys.join('\u0000')
  useEffect(() => {
    if (!unique.data || unique.data.field !== layer.style.uniqueField) return
    const next = fillCategoryColors(layer.style.categoryColors, uniqueKeys)
    if (Object.keys(next).length !== Object.keys(layer.style.categoryColors).length) {
      setLayerStyle(layer.uid, { categoryColors: next })
    }
    // uniqueKeys 用签名参与依赖：数组字面量每次渲染都是新引用。
  }, [uniqueSignature, layer.uid, layer.style.uniqueField, layer.style.categoryColors])

  // 分级：按色带采样出每级色值 + 把断点落到样式（地图侧 step 表达式要它们）。
  useEffect(() => {
    if (layer.style.mode !== 'graduated') return
    const colors = sampleRamp(layer.style.ramp, layer.style.graduatedClasses)
    const breaks = graduated.data?.breaks ?? []
    const colorsStale = layer.style.graduatedColors.length !== layer.style.graduatedClasses
    const breaksStale =
      breaks.length > 1 && layer.style.graduatedBreaks.join(',') !== breaks.join(',')
    if (!colorsStale && !breaksStale) return
    setLayerStyle(layer.uid, {
      graduatedColors: colorsStale ? colors : layer.style.graduatedColors,
      graduatedBreaks: breaks.length > 1 ? breaks : layer.style.graduatedBreaks,
    })
  }, [
    layer.style.mode,
    layer.style.ramp,
    layer.style.graduatedClasses,
    layer.style.graduatedColors,
    layer.style.graduatedBreaks,
    layer.uid,
    graduated.data?.breaks,
    setLayerStyle,
  ])

  const fallbackFill = style.fillColor ?? categoryColor(style.colorIndex)
  const fallbackStroke = style.strokeColor ?? fallbackFill

  return (
    <>
      <Section title="渲染器">
        <Row label="类型">
          <SelectRow
            value={style.mode}
            onChange={(value) => setLayerStyle(layer.uid, { mode: value as RendererMode })}
            options={[
              { value: 'single', label: '单一符号' },
              { value: 'unique', label: '唯一值' },
              { value: 'graduated', label: '分级色彩（分位数）' },
            ]}
          />
        </Row>
        {style.mode === 'unique' ? (
          <Row label="字段">
            <SelectRow
              value={style.uniqueField ?? ''}
              onChange={(value) =>
                setLayerStyle(layer.uid, { uniqueField: value || null, categoryColors: {} })
              }
              title="类目来自 /api/catalog/distinct"
              options={[
                { value: '', label: '选择字段…' },
                ...fields.map((field) => ({
                  value: field.name,
                  label: field.alias || field.name,
                })),
              ]}
            />
          </Row>
        ) : null}
        {style.mode === 'graduated' ? (
          <>
            <Row label="字段">
              <SelectRow
                value={style.graduatedField ?? ''}
                onChange={(value) => setLayerStyle(layer.uid, { graduatedField: value || null })}
                title="只列数值字段"
                options={[
                  { value: '', label: '选择数值字段…' },
                  ...numericFields.map((name) => ({ value: name, label: name })),
                ]}
              />
            </Row>
            <Row label="级数">
              <SelectRow
                value={String(style.graduatedClasses)}
                onChange={(value) =>
                  setLayerStyle(layer.uid, { graduatedClasses: Number(value) })
                }
                options={[3, 5, 7].map((count) => ({ value: String(count), label: `${count} 级` }))}
              />
            </Row>
            <Row label="分级方法">
              <span className="text-[11px] text-foreground">分位数（quantile）</span>
            </Row>
            <Row label="色带">
              <SelectRow
                value={style.ramp}
                onChange={(value) => setLayerStyle(layer.uid, { ramp: value })}
                options={RAMPS.map((ramp) => ({ value: ramp.key, label: ramp.label }))}
              />
            </Row>
            <RampBar colors={sampleRamp(style.ramp, style.graduatedClasses)} />
          </>
        ) : null}
      </Section>

      {style.mode === 'unique' ? (
        <Section title="类目与颜色">
          <UniqueCategories
            result={unique.data}
            loading={unique.isLoading}
            error={unique.error as Error | null}
            field={style.uniqueField}
            colors={style.categoryColors}
            onColor={(key, color) => useGisStore.getState().setCategoryColor(layer.uid, key, color)}
          />
        </Section>
      ) : null}

      {style.mode === 'graduated' ? (
        <Section title="分级区间">
          <GraduatedBreaks
            result={graduated.data}
            loading={graduated.isLoading}
            error={graduated.error as Error | null}
            colors={sampleRamp(style.ramp, style.graduatedClasses)}
          />
        </Section>
      ) : null}

      {style.mode === 'single' ? (
        <Section title="填充">
          <Row label="颜色">
            <ColorSwatch
              value={fallbackFill}
              title="填充色"
              onChange={(color) => setLayerStyle(layer.uid, { fillColor: color })}
            />
            {style.fillColor ? (
              <button
                type="button"
                className="text-[10.5px] text-muted-foreground hover:text-foreground"
                onClick={() => setLayerStyle(layer.uid, { fillColor: null })}
              >
                跟色板
              </button>
            ) : null}
          </Row>
          <RangeRow
            label="不透明度"
            min={0.05}
            max={1}
            step={0.05}
            value={style.opacity}
            onChange={(value) => setLayerStyle(layer.uid, { opacity: value })}
          />
        </Section>
      ) : (
        <Section title="显示">
          <RangeRow
            label="不透明度"
            min={0.05}
            max={1}
            step={0.05}
            value={style.opacity}
            onChange={(value) => setLayerStyle(layer.uid, { opacity: value })}
          />
        </Section>
      )}

      <Section title="描边">
        <Row label="颜色">
          <ColorSwatch
            value={fallbackStroke}
            title="描边色"
            onChange={(color) => setLayerStyle(layer.uid, { strokeColor: color })}
          />
          {style.strokeColor ? (
            <button
              type="button"
              className="text-[10.5px] text-muted-foreground hover:text-foreground"
              onClick={() => setLayerStyle(layer.uid, { strokeColor: null })}
            >
              跟填充
            </button>
          ) : null}
        </Row>
        <RangeRow
          label="线宽"
          min={0}
          max={4}
          step={0.1}
          value={style.lineWidth}
          suffix=" px"
          onChange={(value) => setLayerStyle(layer.uid, { lineWidth: value })}
        />
        <Hint>线型（虚线）本期留空：MapLibre 的 line-dasharray 要按符号层做，先不给假按钮。</Hint>
      </Section>

      {isPoint ? (
        <Section title="标记">
          <RangeRow
            label="大小"
            min={1.5}
            max={10}
            step={0.5}
            value={style.pointSize}
            suffix=" px"
            onChange={(value) => setLayerStyle(layer.uid, { pointSize: value })}
          />
          <Hint>3D 拉伸 / 图标符号本期不做。</Hint>
        </Section>
      ) : null}

      <div className="mt-1 flex gap-1.5">
        <button
          type="button"
          disabled
          title="本期留空：样式持久化需要后端接口"
          className="h-[24px] flex-1 cursor-not-allowed rounded-[var(--gis-radius)] border border-border bg-muted text-[11.5px] text-muted-foreground"
        >
          保存样式
        </button>
        <button
          type="button"
          onClick={() => resetLayerStyle(layer.uid)}
          className="h-[24px] rounded-[var(--gis-radius)] border border-border bg-card px-2 text-[11.5px] text-foreground/80 hover:bg-accent"
        >
          重置
        </button>
      </div>
      <Hint>
        <b className="text-warning">待补</b> 样式持久化（layer style 存储接口未提供）：本期所有改动刷新即丢。
      </Hint>
    </>
  )
}

function UniqueCategories({
  result,
  loading,
  error,
  field,
  colors,
  onColor,
}: {
  result?: DistinctResult
  loading: boolean
  error: Error | null
  field: string | null
  colors: Record<string, string>
  onColor: (key: string, color: string) => void
}) {
  if (!field) {
    return <Hint>先选一个字段；类目值来自 /api/catalog/distinct（不是字段样本）。</Hint>
  }
  if (loading) {
    return (
      <div className="flex items-center gap-1.5 py-2 text-[10.5px] text-muted-foreground">
        <Loader2 className="size-3 animate-spin" /> 正在读取 {field} 的唯一值…
      </div>
    )
  }
  if (error) {
    const issue = describeApiError(error)
    return <Unavailable title={issue.title} reason={`${issue.reason}（字段 ${field}）`} />
  }
  if (!result) return null
  if (result.values.length === 0) {
    return (
      <Unavailable
        title="接口没有返回类目"
        reason={`${field} 的唯一值清单为空（空值 ${result.null_count} 行）——不拿空表冒充类目。`}
      />
    )
  }

  return (
    <>
      <div className="max-h-[236px] overflow-auto pr-0.5">
        {result.values.map((entry, index) => {
          const key = String(entry.value)
          return (
            <div key={key} className="flex items-center gap-[7px] py-[2.5px] text-[11.5px]">
              <ColorSwatch
                size="small"
                value={colors[key] ?? categoryColor(index)}
                title={key}
                onChange={(color) => onColor(key, color)}
              />
              <span className="min-w-0 flex-1 truncate" title={key}>
                {entry.value === null || entry.value === '' ? '（空值）' : key}
              </span>
              <span className="shrink-0 text-[10px] tabular-nums text-muted-foreground">
                {formatNumber(entry.count)}
              </span>
            </div>
          )
        })}
      </div>
      <Hint>
        共 {formatNumber(result.total_distinct)} 个唯一值
        {result.null_count > 0 ? ` · 空值 ${formatNumber(result.null_count)} 行` : ''}
        {result.truncated ? ` · 只列了前 ${result.values.length} 个（limit=200）` : ''}
        {' · '}
        服务端 {Math.round(result.elapsed_ms)} ms
        {result.cache?.hit ? '（命中行缓存）' : ''}
      </Hint>
    </>
  )
}

function GraduatedBreaks({
  result,
  loading,
  error,
  colors,
}: {
  result?: DistinctResult
  loading: boolean
  error: Error | null
  colors: string[]
}) {
  if (loading) {
    return (
      <div className="flex items-center gap-1.5 py-2 text-[10.5px] text-muted-foreground">
        <Loader2 className="size-3 animate-spin" /> 正在算分位断点…
      </div>
    )
  }
  if (error) {
    const issue = describeApiError(error)
    return <Unavailable title={issue.title} reason={issue.reason} />
  }
  if (!result) return <Hint>选一个数值字段后按分位数算断点。</Hint>
  const breaks = result.breaks ?? []
  const counts = result.counts ?? []
  if (breaks.length < 2 || counts.length === 0) {
    return (
      <Unavailable
        title="算不出分级断点"
        reason={result.break_basis?.reason ?? '接口没有返回断点（字段可能不是数值列）'}
      />
    )
  }

  return (
    <>
      <table className="w-full border-collapse text-[10.5px]">
        <thead>
          <tr className="text-muted-foreground">
            <th className="h-[20px] w-[22px] border-b border-border px-1 text-left font-semibold">级</th>
            <th className="h-[20px] border-b border-border px-1 text-left font-semibold">区间</th>
            <th className="h-[20px] w-[46px] border-b border-border px-1 text-right font-semibold">
              要素数
            </th>
          </tr>
        </thead>
        <tbody>
          {counts.map((count, index) => (
            <tr key={index}>
              <td className="h-[21px] border-b border-border/60 px-1">
                <span
                  className="inline-block size-[10px] rounded-[2px] border border-foreground/20 align-middle"
                  style={{ background: colors[index] ?? '#999' }}
                />
              </td>
              <td className="h-[21px] border-b border-border/60 px-1 tabular-nums">
                {formatBreak(breaks[index])} – {formatBreak(breaks[index + 1])}
              </td>
              <td className="h-[21px] border-b border-border/60 px-1 text-right tabular-nums text-muted-foreground">
                {formatNumber(count)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <Hint>
        断点来自 /api/catalog/distinct 的 quantile（{formatNumber(result.break_basis?.values)}{' '}
        个数值
        {result.break_basis?.sampled ? '，抽样' : '，全量'}
        ）；首尾是真实 min / max。
      </Hint>
    </>
  )
}

function formatBreak(value: number | undefined): string {
  if (value === undefined || value === null || !Number.isFinite(value)) return '—'
  if (Math.abs(value) >= 1000) return value.toFixed(0)
  if (Math.abs(value) >= 1) return value.toFixed(2)
  return value.toFixed(4)
}

// ------------------------------------------------------------------ 栅格
function RasterSymbology({ layer, profile }: { layer: GisLayer; profile?: DatasetProfile }) {
  const setLayerStyle = useGisStore((state) => state.setLayerStyle)
  const style = layer.style
  const bandCount =
    profile?.bands_info?.band_count ??
    (profile?.raster as { band_count?: number } | undefined)?.band_count ??
    1
  const bands = Array.from({ length: Math.max(1, Number(bandCount) || 1) }, (_, i) => i + 1)
  const ramp = rampOf(style.ramp)
  const hasRgb = bands.length >= 3

  return (
    <>
      <Section title="波段合成" badge={<Todo />}>
        {(['bandR', 'bandG', 'bandB'] as const).map((key, index) => (
          <Row key={key} label={['红 (R)', '绿 (G)', '蓝 (B)'][index]}>
            <SelectRow
              value={String(style[key])}
              title={hasRgb ? undefined : '该栅格不足 3 段，合成只能重复波段'}
              onChange={(value) => setLayerStyle(layer.uid, { [key]: Number(value) })}
              options={bands.map((band) => ({ value: String(band), label: `Band ${band}` }))}
            />
          </Row>
        ))}
        <Hint>
          波段数 {bandCount}（catalog 扫描值）。波段名 / 波长 / 数据类型看「元数据」的波段明细；
          缺字段的地方标「待补 + 原因」，不编数值。
        </Hint>
      </Section>

      <Section title="拉伸">
        <Row label="类型">
          <SelectRow
            value={style.stretch}
            onChange={(value) => setLayerStyle(layer.uid, { stretch: value as RasterStretch })}
            options={[
              { value: 'minmax', label: '最小–最大' },
              { value: 'std', label: '标准差 (2σ)' },
              { value: 'pct', label: '百分比裁剪 (2%)' },
            ]}
          />
        </Row>
        <RangeRow
          label="伽马"
          min={0.5}
          max={2}
          step={0.1}
          value={style.gamma}
          onChange={(value) => setLayerStyle(layer.uid, { gamma: value })}
        />
      </Section>

      <Section title="色带">
        <Row label="配色">
          <SelectRow
            value={style.ramp}
            onChange={(value) => setLayerStyle(layer.uid, { ramp: value })}
            options={RAMPS.map((item) => ({ value: item.key, label: item.label }))}
          />
        </Row>
        <RampBar colors={ramp.colors} />
        <Hint>色带只作用于显示，不改数据。</Hint>
      </Section>

      <Section title="显示">
        <RangeRow
          label="不透明度"
          min={0.05}
          max={1}
          step={0.05}
          value={style.opacity}
          onChange={(value) => setLayerStyle(layer.uid, { opacity: value })}
        />
        <Hint>栅格不能改单色填充：像素颜色由波段 + 拉伸 + 色带决定。</Hint>
      </Section>

      <Unavailable
        title="栅格影像渲染待后端补"
        reason="/api/catalog/preview 对栅格明确 422（仅支持矢量），没有瓦片或降采样 PNG 接口，所以地图上画不出影像。上面这些选择已记进图层样式，等渲染接口到位就能直接生效；当前改动不会在图上产生可见变化，也不拿假地形冒充。"
      />
    </>
  )
}
