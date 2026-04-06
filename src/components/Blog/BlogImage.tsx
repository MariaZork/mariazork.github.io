import NextImage from 'next/image';
import { useState } from 'react';

interface BlogImageProps {
  /** Image src — relative to /public or absolute URL */
  src: string;
  /** Alt text — required for accessibility */
  alt: string;
  /** Optional caption rendered below the image */
  caption?: string;
  /** Optional explicit width (px). If omitted, uses responsive fill mode. */
  width?: number;
  /** Optional explicit height (px). Required when width is set. */
  height?: number;
  /** Allow clicking to open full-size in a lightbox overlay */
  zoomable?: boolean;
  /** Tailwind class for max-width. Default: 'max-w-full' */
  maxWidth?: string;
  /** Aspect ratio for responsive images (default: '4/3'). Ignored if width/height are set. */
  aspectRatio?: string;
  /** Compact mode with reduced spacing (default: false) */
  compact?: boolean;
}

/**
 * BlogImage — drop-in image component for MDX posts.
 *
 * Usage in MDX:
 *
 *   <BlogImage
 *     src="/images/2022-09-01-.../graph.png"
 *     alt="GCN architecture"
 *     caption="Figure 1: Graph Convolutional Network layer propagation."
 *     zoomable
 *   />
 *
 * For sized images (e.g. small diagrams):
 *
 *   <BlogImage
 *     src="/images/..."
 *     alt="confusion matrix"
 *     width={480}
 *     height={320}
 *     caption="Confusion matrix on test set."
 *   />
 */
export default function BlogImage({
  src,
  alt,
  caption,
  width,
  height,
  zoomable = false,
  maxWidth = 'max-w-4xl',
  aspectRatio = '4/3',
  compact = false,
}: BlogImageProps) {
  const [lightboxOpen, setLightboxOpen] = useState(false);
  const [imgLoaded, setImgLoaded] = useState(false);
  const sized = width !== undefined && height !== undefined;
  const isExternal = src.startsWith('http://') || src.startsWith('https://');

  const imgElement = sized ? (
    <NextImage
      src={src}
      alt={alt}
      width={width}
      height={height}
      className={`rounded-lg object-contain ${zoomable ? 'cursor-zoom-in' : ''}`}
      loading="lazy"
      onLoadingComplete={() => setImgLoaded(true)}
      onClick={zoomable ? () => setLightboxOpen(true) : undefined}
    />
) : (
  <div className="relative w-full">
    <NextImage
      src={src}
      alt={alt}
      width={0}
      height={0}
      sizes="(max-width: 768px) 100vw, (max-width: 1200px) 80vw, 720px"
      className={`w-full h-auto rounded-lg object-contain ${zoomable ? 'cursor-zoom-in' : ''} ${imgLoaded ? 'opacity-100' : 'opacity-0'} transition-opacity duration-300`}
      loading="lazy"
      onLoadingComplete={() => setImgLoaded(true)}
      onClick={zoomable ? () => setLightboxOpen(true) : undefined}
      unoptimized={isExternal}
    />
  </div>
);

  return (
    <>
      <span className={`${maxWidth} mx-auto block`} role="img" aria-label={alt}>
        <span
          className={`block overflow-hidden rounded-lg border border-border shadow-sm bg-surface-alt
            ${zoomable ? 'hover:shadow-card-hover transition-shadow duration-200' : ''}`}
        >
          <span className="relative block w-full">
            {imgElement}
          </span>
        </span>

        {caption && (
          <span className={`block ${compact ? 'mt-1.5 text-[11px]' : 'mt-2 text-xs'} text-center text-ink-muted leading-relaxed px-3 italic`}>
            {caption}
          </span>
        )}
      </span>

      {/* ── Lightbox ── */}
      {zoomable && lightboxOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4"
          onClick={() => setLightboxOpen(false)}
          role="dialog"
          aria-modal="true"
          aria-label={`Full size: ${alt}`}
        >
          {/* Close button */}
          <button
            className="absolute top-4 right-4 w-9 h-9 flex items-center justify-center rounded-full bg-white/10 text-white hover:bg-white/20 transition-colors"
            onClick={() => setLightboxOpen(false)}
            aria-label="Close"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>

          {/* Full-size image */}
          <div
            className="relative max-w-5xl max-h-[90vh] w-full"
            onClick={e => e.stopPropagation()}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={src}
              alt={alt}
              className="w-full h-auto max-h-[85vh] object-contain rounded-xl cursor-zoom-out"
              onClick={() => setLightboxOpen(false)}
            />
            {caption && (
              <p className="mt-3 text-center text-xs text-white/70 italic">{caption}</p>
            )}
          </div>
        </div>
      )}
    </>
  );
}