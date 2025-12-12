import type { Config } from 'tailwindcss';

const config: Config = {
  darkMode: 'class',
  content: [
    './pages/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './app/**/*.{ts,tsx}',
    './src/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        // AgniVed Brand Colors
        brand: {
          primary: '#3b82f6', // blue-600
          secondary: '#10b981', // emerald-500
          accent: '#f59e0b', // amber-500
        },
        // Land Cover Classes
        landcover: {
          water: '#3b82f6',
          trees: '#16a34a',
          grass: '#84cc16',
          flooded: '#06b6d4',
          crops: '#eab308',
          shrub: '#a3e635',
          built: '#f97316',
          bare: '#78716c',
          snow: '#e0f2fe',
        },
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
    },
  },
  plugins: [require('tailwindcss-animate')],
};

export default config;