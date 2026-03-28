import Link from 'next/link';
import Image from 'next/image';
import { useState } from 'react';
import { format } from 'date-fns';
import { Post } from '@/types/post';

interface PostCardProps {
  post: Post;
}

const categoryColors: Record<string, { bg: string; text: string }> = {
  'machine learning': { bg: 'bg-primary-soft', text: 'text-primary' },
  'machine learning theory': { bg: 'bg-primary-soft', text: 'text-primary' },
  'deep learning': { bg: 'bg-secondary-soft', text: 'text-secondary' },
  'cybersecurity': { bg: 'bg-red-50', text: 'text-red-600' },
  'default': { bg: 'bg-surface-alt', text: 'text-ink-muted' },
};

function getCategoryStyle(cat: string) {
  return categoryColors[cat.toLowerCase()] ?? categoryColors['default'];
}

function hasRealImage(img: string | undefined): boolean {
  if (!img) return false;
  if (img === '/images/default-og-image.jpg') return false;
  if (img === '/images/sample_feature_img.png') return false;
  return true;
}

export default function PostCard({ post }: PostCardProps) {
  const [imgError, setImgError] = useState(false);
  const showImage = hasRealImage(post.image) && !imgError;

  return (
    <article className="post-card bg-white rounded-2xl border border-border overflow-hidden flex flex-col h-full shadow-card">

      {/* Image / Fallback */}
      <Link href={`/blog/${post.slug}`} className="block relative h-48 bg-surface-alt overflow-hidden flex-shrink-0 group">
        {showImage ? (
          <Image
            src={post.image}
            alt={post.title}
            fill
            sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
            className="object-cover transition-transform duration-500 group-hover:scale-105"
            onError={() => setImgError(true)}
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-primary/6 via-surface-alt to-secondary/6">
            <span
              className="font-serif font-bold select-none"
              style={{ fontSize: '7rem', lineHeight: 1, color: 'rgba(26,188,156,0.13)' }}
            >
              {post.title.charAt(0).toUpperCase()}
            </span>
            <div className="absolute top-0 right-0 w-20 h-20 bg-gradient-to-bl from-primary/8 to-transparent rounded-bl-3xl" />
          </div>
        )}

        {showImage && (
          <div className="absolute bottom-2 right-2 flex items-center gap-1 bg-black/50 text-white text-[10px] font-medium px-2 py-0.5 rounded-full backdrop-blur-sm">
            <svg className="w-2.5 h-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            {post.readingTime}
          </div>
        )}
      </Link>

      {/* Body */}
      <div className="p-5 flex flex-col flex-grow">

        <div className="flex flex-wrap gap-1.5 mb-3">
          {post.categories.slice(0, 2).map(cat => {
            const style = getCategoryStyle(cat);
            return (
              <Link
                key={cat}
                href={`/categories/${cat.toLowerCase().replace(/\s+/g, '-')}`}
                className={`text-xs font-semibold px-2.5 py-1 rounded-full tag-pill ${style.bg} ${style.text} hover:opacity-75 transition-opacity`}
              >
                {cat}
              </Link>
            );
          })}
        </div>

        <Link href={`/blog/${post.slug}`} className="group block mb-1">
          <h3 className="font-serif text-[1.1rem] font-bold text-ink leading-snug group-hover:text-primary transition-colors">
            {post.title}
          </h3>
        </Link>

        {post.subtitle && (
          <p className="text-xs text-ink-muted italic mb-2 leading-snug">{post.subtitle}</p>
        )}

        <p className="text-xs text-ink-muted leading-relaxed line-clamp-3 mb-4 flex-grow">
          {post.excerpt}
        </p>

        <div className="flex items-center justify-between pt-3 border-t border-border">
          <time className="text-xs text-ink-muted">
            {format(new Date(post.date), 'MMM d, yyyy')}
          </time>
          {!showImage && (
            <span className="text-xs text-ink-muted flex items-center gap-1">
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              {post.readingTime}
            </span>
          )}
        </div>
      </div>
    </article>
  );
}
