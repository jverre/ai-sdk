import nextra from 'nextra'
 
const withNextra = nextra({
  theme: 'nextra-theme-docs',
  themeConfig: './theme.config.jsx'
})
 
// You can include other Next.js configuration options here, in addition to Nextra settings:
export default withNextra({
  output: 'export',
  images: {
    unoptimized: true,
  },
  basePath: process.env.NODE_ENV === 'production' ? '/ai-sdk' : '',
  // This setting is required for static exports
  distDir: 'dist'
})