import React from 'react';
import classnames from 'classnames';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './styles.module.css';

const features = [
  {
    title: <>Multi-channel attribution problem</>,
    imageUrl: 'img/attribution.png',
    description: (
      <>
        Are you using a variety of online marketing channels to reach your customers? Do you want to know the degree each channel contributes to your marketing success? 
      </>
    ),
  },

  {
    title: <>Markov Model</>,
    imageUrl: 'img/markov.png',
    description: (
      <>
        Markov model is a probabilistic model widely adopted to model changing systems. It is perfect to model customers journey though marketing channels. It is a scientific and objective way to calculate how your marketing channels contribute to your revenues. 
      </>
    ),
  },
  {
    title: <>ChannelAttribution</>,
    imageUrl: 'img/logo2.png',
    description: (
      <>
        <i>ChannelAttribution</i> is an open-source project which let you to estimate Markov model from customer journey data easily and quickly. Read our <a href="pdf/ChannelAttributionWhitePaper.pdf">Whitepaper</a>.
      </>
    ),
  },
  
  {
    title: <>Powered by C++</>,
    imageUrl: 'img/cpp.png',
    description: (
      <>
        The core of <i>ChannelAttribution</i> is written in C++, to allow implementing high-performing algorithms.
      </>
    ),
  }, 
  {
    title: <>R-package</>,
    imageUrl: 'img/R.png',
    description: (
      <>
        R-package <i>ChannelAttribution</i> can be downloaded from CRAN.
      </>
    ),
  },
  {
    title: <>Python library</>,
    imageUrl: 'img/python.png',
    description: (
      <>
        Python library <i>ChannelAttribution</i> is available on PyPI repository.
      </>
    ),
  },

 
];

function Feature({imageUrl, title, description}) {
  const imgUrl = useBaseUrl(imageUrl);
  return (
    <div className={classnames('col col--4', styles.feature)}>
      {imgUrl && (
        <div className="text--center">
          <img className={styles.featureImage} src={imgUrl} alt={title} />
        </div>
      )}
      <h3>{title}</h3>
      <p>{description}</p>
    </div>
  );
}

function Home() {
  const context = useDocusaurusContext();
  const {siteConfig = {}} = context;
  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="Description will go into a meta tag in <head />">
      <header className={classnames('hero hero--primary', styles.heroBanner)}>
        <div className="container">
          <h1 className="hero__title">{siteConfig.title}</h1>
          <p className="hero__subtitle">{siteConfig.tagline}</p>
          <div className={styles.buttons}>
            <Link
              className={classnames(
                'button button--outline button--secondary button--lg',
                styles.getStarted,
              )}
              to={useBaseUrl('docs/gettingstarted')}>
              Get Started
            </Link>
          </div>
        </div>
      </header>
      <main>
        {features && features.length > 0 && (
          <section className={styles.features}>
            <div className="container">
              <div className="row">
                {features.map((props, idx) => (
                  <Feature key={idx} {...props} />
                ))}
              </div>
            </div>
          </section>
        )}
      </main>
    </Layout>
  );
}

export default Home;
