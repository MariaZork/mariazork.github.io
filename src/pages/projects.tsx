import React from 'react';
import Layout from '@/components/Layout/Layout';
import Image from 'next/image';
import Link from 'next/link';

interface Project {
  title: string;
  description: string;
  tags: string[];
  href?: string;
  image?: string;
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
  'Kaggle': 'bg-[#20BEFF]/10 text-[#20BEFF] border-[#20BEFF]/20',
  'Live Demo': 'bg-primary-soft text-primary border-primary/20',
  'Research': 'bg-secondary-soft text-secondary border-secondary/20',
};

function ProjectCard({ project }: { project: Project }) {
  const badge = project.badge || '';
  const badgeClass = badgeColors[badge] || 'bg-surface-alt text-ink-muted border-border';

  const inner = (
    <div className="post-card bg-white rounded-2xl border border-border shadow-card p-6 flex flex-col h-full group">
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
      <a href={project.href} target="_blank" rel="noopener noreferrer">
        {inner}
      </a>
    ) : (
      <Link href={project.href}>{inner}</Link>
    );
  }
  return <div>{inner}</div>;
}

function SectionHeader({ label, title }: { label: string; title: string }) {
  return (
    <div className="mb-8">
      <p className="text-xs font-semibold uppercase tracking-widest text-primary mb-1">{label}</p>
      <h2 className="font-serif text-2xl text-ink">{title}</h2>
    </div>
  );
}

export default function ProjectsPage() {
  return (
    <Layout title="Projects — Maria Zorkaltseva" description="Machine learning projects: Kaggle competitions, web applications, and research.">
      {/* Header */}
      <div className="border-b border-border bg-surface-alt">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-14">
          <p className="text-xs font-semibold uppercase tracking-widest text-primary mb-2">Portfolio</p>
          <h1 className="font-serif text-4xl md:text-5xl text-ink mb-4">Projects</h1>
          <p className="text-ink-muted max-w-xl leading-relaxed">
            A collection of Kaggle competition notebooks, deployed web apps, and research reproductions in ML and deep learning.
          </p>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-14 space-y-16">

        {/* Kaggle */}
        <section>
          <SectionHeader label="Competitions" title="Kaggle Projects" />
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {kaggleProjects.map(p => <ProjectCard key={p.title} project={p} />)}
          </div>
        </section>

        {/* Web Apps */}
        <section>
          <SectionHeader label="Deployed" title="Web Applications" />
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {webProjects.map(p => <ProjectCard key={p.title} project={p} />)}
          </div>
        </section>

        {/* Research */}
        <section>
          <SectionHeader label="Experiments" title="Research & Tutorials" />
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {researchProjects.map(p => <ProjectCard key={p.title} project={p} />)}
          </div>
        </section>

      </div>
    </Layout>
  );
}
