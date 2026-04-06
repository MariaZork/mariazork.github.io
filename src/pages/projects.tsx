import React from 'react';
import Layout from '@/components/Layout/Layout';
import Link from 'next/link';
import { useRef, useState } from 'react';

interface Project {
  title: string;
  description: string;
  tags: string[];
  href?: string;
  badge?: string;
}

const kaggleProjects: Project[] = [
  {
    title: 'Images Tiles Generator from WSI',
    description:
      'Preprocessing pipeline to extract tiles from Whole Slide Images (WSI) for the HuBMAP Kaggle challenge.',
    tags: ['Computer Vision', 'WSI', 'Preprocessing', 'Segmentation'],
    href: 'https://www.kaggle.com/mariazorkaltseva/hubmap-train-test-patches-generation',
    badge: 'Kaggle',
  },
  {
    title: 'Glomeruli Segmentation — SeResNext50 UNet',
    description:
      'UNET model with SeResNext-50_32x4d encoder trained with Dice loss for semantic segmentation of Glomeruli areas in kidney tissues.',
    tags: ['PyTorch', 'UNet', 'SeResNeXt', 'Medical Imaging'],
    href: 'https://www.kaggle.com/mariazorkaltseva/hubmap-seresnext50-unet-dice-loss',
    badge: 'Kaggle',
  },
  {
    title: 'Lung Anomaly Detection — Faster R-CNN',
    description:
      'Faster R-CNN trained with IceVision for detection of anomalies in chest X-rays from the VinBigData dataset.',
    tags: ['Object Detection', 'Faster R-CNN', 'IceVision', 'Radiology'],
    href: 'https://www.kaggle.com/mariazorkaltseva/vinbigdata-eda-faster-rcnn-icevision-training',
    badge: 'Kaggle',
  },
  {
    title: 'Audio Signal Processing - EDA and Feature Engineering',
    description:
      'Exploratory data analysis and feature engineering for audio signal processing tasks.',
    tags: ['Audio Processing', 'EDA', 'Feature Engineering'],
    href: 'https://www.kaggle.com/code/mariazorkaltseva/audio-birdclef-2024-eda-and-feature-engineering',
    badge: 'Kaggle',
  },
];

const webProjects: Project[] = [
  {
    title: 'Phishing URL Detector',
    description:
      'Web application for real-time phishing URL detection using a Random Forest classifier trained on NLP features extracted from URL structure and text patterns.',
    tags: ['Flask', 'Random Forest', 'NLP', 'Cybersecurity'],
    href: 'https://phishing-url-detector.herokuapp.com/',
    badge: 'Live Demo',
  },
];

const openSourceProjects: Project[] = [
  {
    title: 'HistoCoreML',
    description:
      'Production-ready ML framework for computational histology. Covers WSI I/O, patch tiling, Macenko H&E normalisation, TorchScript/ONNX segmentation, foundation model embeddings (UNI, ViT, CTransPath), biomarker extraction, and multi-format output (TIFF, RLE, GeoJSON, Zarr).',
    tags: ['PyTorch', 'ONNX', 'Digital Pathology', 'WSI', 'Segmentation', 'Apache-2.0'],
    href: 'https://github.com/MariaZork/histocore-ml',
    badge: 'Open Source',
  },
];

const researchProjects: Project[] = [
  {
    title: 'Graph Neural Networks for Fraud Detection',
    description:
      'Comparison of GCN and GAT architectures on the Elliptic Bitcoin dataset for illicit transaction classification. GAT improved recall from 0.45 to 0.67.',
    tags: ['PyG', 'GCN', 'GAT', 'Fraud Detection', 'Crypto'],
    href: '/blog/2022-09-01-graph-neural-networks-for-fraud-detection-in-crypto-transactions',
    badge: 'Research',
  },
  {
    title: 'Colorectal Histology Classification',
    description:
      'ResNet-50 trained from scratch on the Colorectal Histology MNIST dataset achieving 92% accuracy across 8 tissue types.',
    tags: ['ResNet', 'PyTorch', 'Digital Pathology', 'Classification'],
    href: '/blog/2020-10-26-colorectal-tissue-classification',
    badge: 'Research',
  },
  {
    title: 'Metro Traffic Forecasting — LSTM',
    description:
      'Comparative study of 1-layer, 2-layer, and bidirectional LSTM architectures for multivariate multi-step traffic volume prediction.',
    tags: ['LSTM', 'PyTorch', 'Time Series', 'Forecasting'],
    href: '/blog/2020-06-20-metro-traffic-forecasting',
    badge: 'Research',
  },
];

const badgeColors: Record<string, string> = {
  Kaggle: 'bg-[#20BEFF]/10 text-[#20BEFF] border-[#20BEFF]/20',
  'Live Demo': 'bg-primary-soft text-primary border-primary/20',
  Research: 'bg-secondary-soft text-secondary border-secondary/20',
  'Open Source': 'bg-emerald-50 text-emerald-700 border-emerald-200',
};

function ProjectCard({ project }: { project: Project }) {
  const badge = project.badge || '';
  const badgeClass = badgeColors[badge] || 'bg-surface-alt text-ink-muted border-border';

  const inner = (
    <div className="post-card bg-white rounded-2xl border border-border shadow-card p-6 flex flex-col h-full group select-none">
      <div className="flex items-start justify-between gap-3 mb-4">
        <div className="w-10 h-10 rounded-xl bg-surface-alt border border-border flex items-center justify-center flex-shrink-0">
          <svg className="w-5 h-5 text-ink-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
          </svg>
        </div>
        {badge && (
          <span className={`text-xs font-semibold px-2.5 py-1 rounded-full border ${badgeClass}`}>
            {badge}
          </span>
        )}
      </div>
      <h3 className="font-serif text-lg text-ink group-hover:text-primary transition-colors mb-2 leading-snug">
        {project.title}
      </h3>
      <p className="text-sm text-ink-muted leading-relaxed mb-5 flex-grow">
        {project.description}
      </p>
      <div className="flex flex-wrap gap-1.5 mt-auto pt-4 border-t border-border">
        {project.tags.map(tag => (
          <span key={tag} className="text-xs px-2 py-0.5 rounded bg-surface-alt text-ink-muted">
            {tag}
          </span>
        ))}
      </div>
    </div>
  );

  if (project.href) {
    const isExternal = project.href.startsWith('http');
    return isExternal ? (
      <a href={project.href} target="_blank" rel="noopener noreferrer" className="block h-full">
        {inner}
      </a>
    ) : (
      <Link href={project.href} className="block h-full">
        {inner}
      </Link>
    );
  }
  return <div className="h-full">{inner}</div>;
}

// ── Carousel (universal) ─────────────────────────────────────────
function ProjectCarousel({ projects }: { projects: Project[] }) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [current, setCurrent] = useState(0);

  // Cards visible = 1 on mobile, 2 on md, 3 on lg
  // We control via scroll snap + index tracking
  const cardWidth = 340; // approximate, used for button scroll amount
  const gap = 24;

  const scrollTo = (idx: number) => {
    const clamped = Math.max(0, Math.min(idx, projects.length - 1));
    setCurrent(clamped);
    trackRef.current?.scrollTo({
      left: clamped * (cardWidth + gap),
      behavior: 'smooth',
    });
  };

  const handleScroll = () => {
    if (!trackRef.current) return;
    const scrollLeft = trackRef.current.scrollLeft;
    const idx = Math.round(scrollLeft / (cardWidth + gap));
    setCurrent(idx);
  };

  return (
    <div className="relative">
      {/* Track */}
      <div
        ref={trackRef}
        onScroll={handleScroll}
        className="flex gap-6 overflow-x-auto pb-4 snap-x snap-mandatory scrollbar-none"
        style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}
      >
        {projects.map(p => (
          <div
            key={p.title}
            className="snap-start flex-shrink-0 w-[85vw] sm:w-[340px]"
            style={{ minHeight: 260 }}
          >
            <ProjectCard project={p} />
          </div>
        ))}
      </div>

      {/* Controls */}
      <div className="flex items-center gap-4 mt-5">
        {/* Dots */}
        <div className="flex gap-1.5">
          {projects.map((_, i) => (
            <button
              key={i}
              onClick={() => scrollTo(i)}
              aria-label={`Go to slide ${i + 1}`}
              className={`h-1.5 rounded-full transition-all duration-300 ${
                i === current
                  ? 'w-5 bg-primary'
                  : 'w-1.5 bg-border hover:bg-ink-muted'
              }`}
            />
          ))}
        </div>

        {/* Arrows */}
        <div className="flex gap-2 ml-auto">
          <button
            onClick={() => scrollTo(current - 1)}
            disabled={current === 0}
            aria-label="Previous"
            className="w-8 h-8 rounded-full border border-border flex items-center justify-center text-ink-muted hover:border-primary hover:text-primary transition disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>
          <button
            onClick={() => scrollTo(current + 1)}
            disabled={current === projects.length - 1}
            aria-label="Next"
            className="w-8 h-8 rounded-full border border-border flex items-center justify-center text-ink-muted hover:border-primary hover:text-primary transition disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}

function SectionHeader({ label, title }: { label: string; title: string }) {
  return (
    <div className="mb-8">
      <p className="text-xs font-semibold uppercase tracking-widest text-primary mb-1">{label}</p>
      <h2 className="font-serif text-2xl text-ink">{title}</h2>
    </div>
  );
}

interface ProjectSectionProps {
  label: string;
  title: string;
  projects: Project[];
  headerAction?: React.ReactNode;
  columns?: 2 | 3;
}

function ProjectSection({ label, title, projects, headerAction, columns = 3 }: ProjectSectionProps) {
  const useCarousel = projects.length > 3;
  const gridClass = columns === 2
    ? 'grid grid-cols-1 md:grid-cols-2 gap-6'
    : 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6';

  return (
    <section>
      <div className="flex items-end justify-between mb-8">
        <SectionHeader label={label} title={title} />
        {headerAction}
      </div>
      {useCarousel
        ? <ProjectCarousel projects={projects} />
        : <div className={gridClass}>{projects.map(p => <ProjectCard key={p.title} project={p} />)}</div>
      }
    </section>
  );
}

export default function ProjectsPage() {
  return (
    <Layout
      title="Projects — Maria Zorkaltseva"
      description="Machine learning projects: Kaggle competitions, web applications, and research."
    >
      {/* Header */}
      <div className="border-b border-border bg-surface-alt">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-14">
          <p className="text-xs font-semibold uppercase tracking-widest text-primary mb-2">Portfolio</p>
          <h1 className="font-serif text-4xl md:text-5xl text-ink mb-4">Projects</h1>
          <p className="text-ink-muted max-w-xl leading-relaxed">
            A collection of Kaggle competition notebooks, deployed web apps, research reproductions, and open source personal projects.
          </p>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-14 space-y-16">

        <ProjectSection
          label="Competitions"
          title="Kaggle Projects"
          projects={kaggleProjects}
          columns={3}
          headerAction={
            <a
              href="https://www.kaggle.com/mariazorkaltseva/code"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs font-medium text-ink-muted hover:text-primary transition-colors flex items-center gap-1 mb-8"
            >
              View all on Kaggle
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
              </svg>
            </a>
          }
        />

        <ProjectSection label="Deployed" title="Web Applications" projects={webProjects} columns={2} />

        <ProjectSection label="Open Source" title="Libraries & Frameworks" projects={openSourceProjects} columns={2} />

        <ProjectSection label="Experiments" title="Research & Tutorials" projects={researchProjects} columns={3} />

      </div>
    </Layout>
  );
}