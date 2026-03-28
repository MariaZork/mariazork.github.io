/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: '#1ABC9C',
        'primary-dark': '#0e9d84',
        'primary-soft': '#e6f9f5',
        secondary: '#6b5ce7',
        'secondary-soft': '#f0eeff',
        accent: '#f59e0b',
        ink: '#1a1a2e',
        'ink-muted': '#4a4a6a',
        surface: '#fafaf8',
        'surface-alt': '#f2f2ee',
        border: '#e4e4de',
        dark: '#1a1a2e',
        light: '#fafaf8',
      },
      fontFamily: {
        sans: ['DM Sans', 'system-ui', 'sans-serif'],
        serif: ['DM Serif Display', 'Georgia', 'serif'],
        heading: ['DM Serif Display', 'Georgia', 'serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      fontSize: {
        '2xs': '0.65rem',
      },
      borderRadius: {
        '2xl': '1rem',
        '3xl': '1.5rem',
      },
      boxShadow: {
        'card': '0 1px 3px rgba(0,0,0,0.05), 0 4px 16px rgba(0,0,0,0.06)',
        'card-hover': '0 4px 12px rgba(0,0,0,0.08), 0 16px 40px rgba(0,0,0,0.1)',
        'soft': '0 2px 8px rgba(0,0,0,0.04)',
      },
      typography: (theme) => ({
        DEFAULT: {
          css: {
            '--tw-prose-body': theme('colors.ink'),
            '--tw-prose-headings': theme('colors.ink'),
            '--tw-prose-lead': theme('colors.ink-muted'),
            '--tw-prose-links': theme('colors.primary'),
            '--tw-prose-bold': theme('colors.ink'),
            '--tw-prose-counters': theme('colors.ink-muted'),
            '--tw-prose-bullets': theme('colors.primary'),
            '--tw-prose-hr': theme('colors.border'),
            '--tw-prose-quotes': theme('colors.ink-muted'),
            '--tw-prose-quote-borders': theme('colors.primary'),
            '--tw-prose-captions': theme('colors.ink-muted'),
            '--tw-prose-code': theme('colors.secondary'),
            '--tw-prose-pre-code': '#e6edf3',
            '--tw-prose-pre-bg': '#0d1117',
            '--tw-prose-th-borders': theme('colors.border'),
            '--tw-prose-td-borders': theme('colors.border'),
          },
        },
      }),
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
};