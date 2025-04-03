import { GitHubIcon } from 'nextra/icons'

export default {
  index: {
    title: 'AI SDK',
    type: 'doc'
  },
  guides: 'Guides',
  providers: 'Providers'
}

// Custom component for italicized text
function Italic({ children, ...props }) {
  return <i {...props}>{children}</i>
} 