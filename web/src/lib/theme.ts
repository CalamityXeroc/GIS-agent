import { create } from 'zustand'

export type Theme = 'light' | 'dark'

/** index.html 的首屏内联脚本读同一个键，保证刷新不闪白。 */
const STORAGE_KEY = 'gis-agent-theme'

function readInitialTheme(): Theme {
  if (typeof window === 'undefined') return 'light'
  try {
    const saved = window.localStorage.getItem(STORAGE_KEY)
    if (saved === 'light' || saved === 'dark') return saved
  } catch {
    /* 隐私模式下 localStorage 可能抛异常，退回系统偏好 */
  }
  return window.matchMedia('(prefers-color-scheme: dark)').matches
    ? 'dark'
    : 'light'
}

function applyTheme(theme: Theme) {
  document.documentElement.classList.toggle('dark', theme === 'dark')
  try {
    window.localStorage.setItem(STORAGE_KEY, theme)
  } catch {
    /* 存不下就只影响下次刷新，本次仍然生效 */
  }
}

interface ThemeState {
  theme: Theme
  setTheme: (theme: Theme) => void
  toggle: () => void
}

export const useThemeStore = create<ThemeState>((set, get) => ({
  theme: readInitialTheme(),
  setTheme: (theme) => {
    applyTheme(theme)
    set({ theme })
  },
  toggle: () => get().setTheme(get().theme === 'dark' ? 'light' : 'dark'),
}))

/** 根组件挂载时同步一次：内联脚本已经切好类，这里只对齐 store。 */
export function syncThemeClass() {
  const { theme } = useThemeStore.getState()
  document.documentElement.classList.toggle('dark', theme === 'dark')
}
