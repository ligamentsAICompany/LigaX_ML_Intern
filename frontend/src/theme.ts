import { createTheme, type ThemeOptions } from '@mui/material/styles';

// ── Shared tokens ────────────────────────────────────────────────
const sharedTypography: ThemeOptions['typography'] = {
  fontFamily: '"Plus Jakarta Sans", system-ui, -apple-system, "Segoe UI", sans-serif',
  fontSize: 15,
  button: {
    fontFamily: '"Plus Jakarta Sans", system-ui, -apple-system, "Segoe UI", sans-serif',
    textTransform: 'none' as const,
    fontWeight: 600,
  },
};

const sharedComponents: ThemeOptions['components'] = {
  MuiButton: {
    styleOverrides: {
      root: {
        borderRadius: '10px',
        fontWeight: 600,
        transition: 'transform 0.06s ease, background 0.12s ease, box-shadow 0.12s ease',
        '&:hover': { transform: 'translateY(-1px)' },
      },
    },
  },
  MuiPaper: {
    styleOverrides: {
      root: { backgroundImage: 'none' },
    },
  },
};

const sharedShape: ThemeOptions['shape'] = { borderRadius: 12 };

// ── Dark palette ─────────────────────────────────────────────────
const darkVars = {
  '--bg': '#020617',
  '--panel': '#0F172A',
  '--surface': '#111C31',
  '--surface-strong': '#1E293B',
  '--text': '#F8FAFC',
  '--muted-text': '#94A3B8',
  '--accent-yellow': '#F59E0B',
  '--accent-yellow-weak': 'rgba(245,158,11,0.12)',
  '--accent-green': '#22C55E',
  '--accent-green-weak': 'rgba(34,197,94,0.12)',
  '--accent-red': '#F87171',
  '--accent-blue': '#38BDF8',
  '--shadow-1': '0 22px 80px rgba(0,0,0,0.45)',
  '--radius-lg': '24px',
  '--radius-md': '14px',
  '--focus': '0 0 0 3px rgba(34,197,94,0.22)',
  '--border': 'rgba(148,163,184,0.16)',
  '--border-hover': 'rgba(34,197,94,0.45)',
  '--code-bg': 'rgba(2,6,23,0.78)',
  '--tool-bg': 'rgba(15,23,42,0.82)',
  '--tool-border': 'rgba(148,163,184,0.16)',
  '--hover-bg': 'rgba(148,163,184,0.08)',
  '--composer-bg': 'rgba(15,23,42,0.82)',
  '--msg-gradient': 'linear-gradient(180deg, rgba(30,41,59,0.72), rgba(15,23,42,0.38))',
  '--body-gradient': 'radial-gradient(circle at 15% 0%, rgba(34,197,94,0.13), transparent 30%), radial-gradient(circle at 85% 12%, rgba(56,189,248,0.08), transparent 34%), linear-gradient(180deg, #020617 0%, #07111F 50%, #020617 100%)',
  '--scrollbar-thumb': '#334155',
  '--success-icon': '#22C55E',
  '--error-icon': '#F87171',
  '--clickable-text': 'rgba(255, 255, 255, 0.9)',
  '--clickable-underline': 'rgba(255,255,255,0.3)',
  '--code-panel-bg': '#020617',
  '--tab-active-bg': 'rgba(34,197,94,0.12)',
  '--tab-active-border': 'rgba(34,197,94,0.22)',
  '--tab-hover-bg': 'rgba(148,163,184,0.08)',
  '--tab-close-hover': 'rgba(255,255,255,0.1)',
  '--plan-bg': 'rgba(15,23,42,0.82)',
} as const;

// ── Light palette ────────────────────────────────────────────────
const lightVars = {
  '--bg': '#FFFFFF',
  '--panel': '#F7F8FA',
  '--surface': '#F0F1F3',
  '--text': '#1A1A2E',
  '--muted-text': '#6B7280',
  '--accent-yellow': '#FF9D00',
  '--accent-yellow-weak': 'rgba(255,157,0,0.08)',
  '--accent-green': '#16A34A',
  '--accent-red': '#DC2626',
  '--shadow-1': '0 4px 12px rgba(0,0,0,0.08)',
  '--radius-lg': '20px',
  '--radius-md': '12px',
  '--focus': '0 0 0 3px rgba(255,157,0,0.15)',
  '--border': 'rgba(0,0,0,0.08)',
  '--border-hover': 'rgba(0,0,0,0.15)',
  '--code-bg': 'rgba(0,0,0,0.04)',
  '--tool-bg': 'rgba(0,0,0,0.03)',
  '--tool-border': 'rgba(0,0,0,0.08)',
  '--hover-bg': 'rgba(0,0,0,0.04)',
  '--composer-bg': 'rgba(0,0,0,0.02)',
  '--msg-gradient': 'linear-gradient(180deg, rgba(0,0,0,0.01), transparent)',
  '--body-gradient': 'linear-gradient(180deg, #FFFFFF, #F7F8FA)',
  '--scrollbar-thumb': '#C4C8CC',
  '--success-icon': '#FF9D00',
  '--error-icon': '#DC2626',
  '--clickable-text': 'rgba(0, 0, 0, 0.85)',
  '--clickable-underline': 'rgba(0,0,0,0.25)',
  '--code-panel-bg': '#F5F6F8',
  '--tab-active-bg': 'rgba(0,0,0,0.06)',
  '--tab-active-border': 'rgba(0,0,0,0.1)',
  '--tab-hover-bg': 'rgba(0,0,0,0.04)',
  '--tab-close-hover': 'rgba(0,0,0,0.08)',
  '--plan-bg': 'rgba(0,0,0,0.03)',
} as const;

// ── Shared CSS baseline (scrollbar, code, brand-logo) ────────────
function makeCssBaseline(vars: Record<string, string>) {
  return {
    styleOverrides: {
      ':root': vars,
      body: {
        background: 'var(--body-gradient)',
        color: 'var(--text)',
        fontFamily: '"Plus Jakarta Sans", system-ui, -apple-system, "Segoe UI", sans-serif',
        scrollbarWidth: 'thin' as const,
        '&::-webkit-scrollbar': { width: '8px', height: '8px' },
        '&::-webkit-scrollbar-thumb': {
          backgroundColor: 'var(--scrollbar-thumb)',
          borderRadius: '2px',
        },
        '&::-webkit-scrollbar-track': { backgroundColor: 'transparent' },
      },
      'code, pre': {
        fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, "Roboto Mono", monospace',
      },
      '.brand-logo': {
        position: 'relative' as const,
        padding: '6px',
        borderRadius: '8px',
        '&::after': {
          content: '""',
          position: 'absolute' as const,
          inset: '-6px',
          borderRadius: '10px',
          background: 'var(--accent-yellow-weak)',
          zIndex: -1,
          pointerEvents: 'none' as const,
        },
      },
    },
  };
}

function makeDrawer() {
  return {
    styleOverrides: {
      paper: {
        backgroundColor: 'var(--panel)',
        borderRight: '1px solid var(--border)',
      },
    },
  };
}

function makeTextField() {
  return {
    styleOverrides: {
      root: {
        '& .MuiOutlinedInput-root': {
          borderRadius: 'var(--radius-md)',
          '& fieldset': { borderColor: 'var(--border)' },
          '&:hover fieldset': { borderColor: 'var(--border-hover)' },
          '&.Mui-focused fieldset': {
            borderColor: 'var(--accent-green)',
            borderWidth: '1px',
            boxShadow: 'var(--focus)',
          },
        },
      },
    },
  };
}

// ── Theme builders ───────────────────────────────────────────────
export const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: '#22C55E', light: '#4ADE80', dark: '#16A34A', contrastText: '#020617' },
    secondary: { main: '#38BDF8' },
    background: { default: '#020617', paper: '#0F172A' },
    text: { primary: '#F8FAFC', secondary: '#94A3B8' },
    divider: 'rgba(148,163,184,0.16)',
    success: { main: '#22C55E' },
    error: { main: '#F87171' },
    warning: { main: '#F59E0B' },
    info: { main: '#58A6FF' },
  },
  typography: sharedTypography,
  components: {
    ...sharedComponents,
    MuiCssBaseline: makeCssBaseline(darkVars),
    MuiDrawer: makeDrawer(),
    MuiTextField: makeTextField(),
  },
  shape: sharedShape,
});

export const lightTheme = createTheme({
  palette: {
    mode: 'light',
    primary: { main: '#FF9D00', light: '#FFB740', dark: '#E08C00', contrastText: '#fff' },
    secondary: { main: '#E08C00' },
    background: { default: '#FFFFFF', paper: '#F7F8FA' },
    text: { primary: '#1A1A2E', secondary: '#6B7280' },
    divider: 'rgba(0,0,0,0.08)',
    success: { main: '#16A34A' },
    error: { main: '#DC2626' },
    warning: { main: '#FF9D00' },
    info: { main: '#2563EB' },
  },
  typography: sharedTypography,
  components: {
    ...sharedComponents,
    MuiCssBaseline: makeCssBaseline(lightVars),
    MuiDrawer: makeDrawer(),
    MuiTextField: makeTextField(),
  },
  shape: sharedShape,
});

// Keep default export for backwards compat
export default darkTheme;
