import { GetStaticProps, GetStaticPaths } from 'next';
import { MDXRemote, MDXRemoteSerializeResult } from 'next-mdx-remote';
import { serialize } from 'next-mdx-remote/serialize';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeHighlight from 'rehype-highlight';
import rehypeSlug from 'rehype-slug';
import rehypeAutolinkHeadings from 'rehype-autolink-headings';
import remarkGfm from 'remark-gfm';
import Link from 'next/link';
import Layout from '@/components/Layout/Layout';
import ReadingProgress from '@/components/Blog/ReadingProgress';
import TableOfContents from '@/components/Blog/TableOfContents';
import SocialShare from '@/components/Blog/SocialShare';
import BackToTop from '@/components/UI/BackToTop';
import SectionDivider from '@/components/Blog/SectionDivider';
import { Post } from '@/types/post';
import { getAllPosts, getPostBySlug } from '@/lib/posts';
import { mdxComponents, CodeCellProvider } from '@/lib/mdx-components';
import { format } from 'date-fns';
import 'highlight.js/styles/github-dark.css';

interface PostPageProps {
  post: Post;
  mdxSource: MDXRemoteSerializeResult;
  headings: Array<{ id: string; text: string; level: number }>;
}

const components = {
  a: (props: any) => (
    <a
      {...props}
      className="text-primary hover:text-primary-dark underline underline-offset-2"
      target={props.href?.startsWith('http') ? '_blank' : undefined}
      rel={props.href?.startsWith('http') ? 'noopener noreferrer' : undefined}
    />
  ),
  SectionDivider,
  ...mdxComponents,
};

export default function PostPage({ post, mdxSource, headings }: PostPageProps) {
  return (
    <Layout
      title={`${post.title}${post.subtitle ? ' — ' + post.subtitle : ''}`}
      description={post.excerpt}
      image={post.image}
      type="article"
      publishedAt={post.date}
      author={post.author}
      keywords={post.tags}
    >
      <ReadingProgress />

      {/* ── Hero ── */}
      <header className="border-b border-border bg-surface-alt">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 py-12 md:py-16">
          {/* Categories */}
          <div className="flex flex-wrap gap-2 mb-5">
            {post.categories.map(cat => (
              <Link
                key={cat}
                href={`/categories/${cat.toLowerCase().replace(/\s+/g, '-')}`}
                className="text-xs font-semibold px-2.5 py-1 rounded-full bg-primary-soft text-primary hover:opacity-80 transition"
              >
                {cat}
              </Link>
            ))}
          </div>

          <h1 className="font-serif text-3xl md:text-4xl text-ink leading-tight mb-3">
            {post.title}
          </h1>

          {post.subtitle && (
            <p className="text-lg text-ink-muted italic mb-6">{post.subtitle}</p>
          )}

          <div className="flex flex-wrap items-center gap-4 text-sm text-ink-muted">
            <span className="font-medium text-ink">{post.author}</span>
            <span className="text-border">·</span>
            <time>{format(new Date(post.date), 'MMMM d, yyyy')}</time>
            <span className="text-border">·</span>
            <span className="flex items-center gap-1">
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              {post.readingTime}
            </span>
          </div>

          {/* Tags */}
          {post.tags.length > 0 && (
            <div className="flex flex-wrap gap-2 mt-6">
              {post.tags.map(tag => (
                <Link
                  key={tag}
                  href={`/tags/${tag.toLowerCase().replace(/\s+/g, '-')}`}
                  className="text-xs px-2.5 py-1 rounded-full bg-white border border-border text-ink-muted hover:border-primary/40 hover:text-primary transition tag-pill"
                >
                  #{tag}
                </Link>
              ))}
            </div>
          )}
        </div>
      </header>

      {/* ── Content ── */}
      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-10">

          {/* Article */}
          <article className="lg:col-span-8">
            <div className="bg-white rounded-2xl border border-border shadow-card p-6 md:p-10 prose prose-lg max-w-none">
              <CodeCellProvider>
                <MDXRemote {...mdxSource} components={components} />
              </CodeCellProvider>
            </div>

            {/* Tags footer */}
            {post.tags.length > 0 && (
              <div className="mt-8 flex flex-wrap gap-2 items-center">
                <span className="text-xs text-ink-muted font-medium uppercase tracking-wider mr-1">Tags:</span>
                {post.tags.map(tag => (
                  <Link
                    key={tag}
                    href={`/tags/${tag.toLowerCase().replace(/\s+/g, '-')}`}
                    className="text-xs px-2.5 py-1 rounded-full bg-white border border-border text-ink-muted hover:border-primary/40 hover:text-primary transition tag-pill"
                  >
                    #{tag}
                  </Link>
                ))}
              </div>
            )}

            {/* Share */}
            <div className="mt-8">
              <SocialShare title={post.title} slug={post.slug} />
            </div>
          </article>

          {/* Sidebar */}
          <aside className="lg:col-span-4">
            <div className="sticky top-20 space-y-6">
              {headings.length > 0 && (
                <TableOfContents headings={headings} />
              )}

              {post.categories.length > 0 && (
                <div className="bg-white rounded-2xl border border-border shadow-card p-5">
                  <h3 className="text-sm font-semibold text-ink-muted uppercase tracking-wider mb-3">
                    Categories
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {post.categories.map(cat => (
                      <Link
                        key={cat}
                        href={`/categories/${cat.toLowerCase().replace(/\s+/g, '-')}`}
                        className="text-xs px-3 py-1.5 rounded-full bg-primary-soft text-primary hover:opacity-80 transition font-medium"
                      >
                        {cat}
                      </Link>
                    ))}
                  </div>
                </div>
              )}

              {/* Back to blog */}
              <Link
                href="/blog"
                className="flex items-center gap-2 text-sm text-ink-muted hover:text-primary transition-colors group"
              >
                <svg className="w-4 h-4 group-hover:-translate-x-0.5 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
                Back to all posts
              </Link>
            </div>
          </aside>
        </div>
      </div>

      <BackToTop />
    </Layout>
  );
}

export const getStaticPaths: GetStaticPaths = async () => {
  const posts = getAllPosts();
  return {
    paths: posts.map(post => ({ params: { slug: post.slug } })),
    fallback: false,
  };
};

export const getStaticProps: GetStaticProps = async ({ params }) => {
  const post = getPostBySlug(params!.slug as string);

  const headingRegex = /^(#{2,4})\s+(.+)$/gm;
  const headings: Array<{ id: string; text: string; level: number }> = [];
  let match;
  while ((match = headingRegex.exec(post.content)) !== null) {
    const level = match[1].length;
    const text = match[2].replace(/[*_`]/g, '');
    const id = text.toLowerCase().replace(/[^\w]+/g, '-');
    headings.push({ id, text, level });
  }

  const mdxSource = await serialize(post.content, {
    parseFrontmatter: false,
    mdxOptions: {
      remarkPlugins: [remarkGfm, remarkMath],
      rehypePlugins: [
        rehypeHighlight,
        rehypeSlug,
        [rehypeAutolinkHeadings, { behavior: 'wrap' }],
        rehypeKatex,
      ],
    },
  });

  return {
    props: {
      post: JSON.parse(JSON.stringify(post)),
      mdxSource,
      headings,
    },
  };
};