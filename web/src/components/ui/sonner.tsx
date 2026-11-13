import { Toaster as SonnerToaster } from 'sonner'

import { useThemeStore } from '@/lib/theme'

/** 操作反馈 toast（§2.4）。跟随当前主题，不用 system，避免深浅不一致。 */
export function Toaster() {
  const theme = useThemeStore((state) => state.theme)
  return (
    <SonnerToaster
      theme={theme}
      position="top-center"
      richColors
      closeButton
      toastOptions={{
        classNames: {
          toast: 'rounded-md border border-border text-sm',
        },
      }}
    />
  )
}
