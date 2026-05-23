import CssBaseline from '@mui/material/CssBaseline';
import { ThemeProvider } from '@mui/material/styles';
import App from './App';
import { useLayoutStore } from './store/layoutStore';
import { darkTheme, lightTheme } from './theme';

export default function Root() {
  const themeMode = useLayoutStore((s) => s.themeMode);
  const theme = themeMode === 'light' ? lightTheme : darkTheme;

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <App />
    </ThemeProvider>
  );
}
