import withMDX from '@next/mdx';

const nextConfig = {
  reactStrictMode: true,
  experimental: {
    mdxRs: true,
  },
  pageExtensions: ['js', 'jsx', 'ts', 'tsx', 'md', 'mdx'],
  output: 'export',
  trailingSlash: true,
  images: {
    unoptimized: true,
  },
};

export default withMDX()(nextConfig);
