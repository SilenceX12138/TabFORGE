<section class="home-banner">
<div class="landing-container banner-inner">
<div class="banner-title">
<h1>TabFORGE</h1><p class="hero-subtitle">Tabular Foundation Model for Generative Modelling</p>
<h2>One foundation model, many downstream tasks</h2>
<p class="hero-description">Generate, predict, embed, and complete mixed-type tables through a shared scikit-learn-style interface.</p>
<p class="hero-meta"><a href="https://github.com/SilenceX12138/TabCamel">Data preparation · TabCamel</a> · <a href="https://github.com/SilenceX12138/TabEval">Evaluation · TabEval</a> · <a href="https://github.com/SilenceX12138/TabStruct">Structural fidelity · TabStruct</a></p>
<div class="hero-actions"><a class="button primary" href="getting_started.md">Getting Started</a><a class="button secondary" href="paper.md">NeurIPS 2026 Paper</a><a class="button secondary" href="https://github.com/SilenceX12138/TabFORGE">Source Code</a></div>
<p class="hero-meta">Install with <code>pip install tabforge-ml</code> · <a href="https://pypi.org/project/tabforge-ml/">PyPI</a></p>
</div>
<figure class="foundation-figure"><a class="hero-figure" href="assets/foundation-model.svg" target="_blank" rel="noopener" aria-label="Open the foundation model figure at full size"><img class="hero-art" src="assets/foundation-model.svg" alt="Mixed-type tables feed a shared TabFORGE foundation model for generation, classification, regression, embedding, imputation, clustering, and anomaly detection" width="1280" height="640"></a></figure>
</div>
</section>

<div class="landing-container task-grid" aria-label="TabFORGE tasks">
<section class="task-card">
<div class="task-card-body"><h3><a href="user_guide.md#generation">Generation</a></h3><p>Create synthetic mixed-type tables with the fitted column schema.</p><p><strong>Applications:</strong> Data augmentation in data-scarce settings, such as small patient cohorts or rare industrial-fault records.<br><strong>Estimator:</strong> <a href="public_estimator_api_estimators.md#tabforgegenerator">TabFORGEGenerator</a></p></div>
<a class="task-art" href="tutorials.md#generation" aria-label="Generation examples"><img src="assets/generation.svg" alt="A table transformed into a synthetic table" width="240" height="120"></a>
<a class="task-examples" href="tutorials.md#generation">Examples</a>
</section>
<section class="task-card">
<div class="task-card-body"><h3><a href="user_guide.md#prediction">Prediction</a></h3><p>Predict class labels or continuous targets with uncertainty summaries.</p><p><strong>Applications:</strong> Classify customer churn or equipment faults; estimate energy use or material properties.<br><strong>Estimator:</strong> <a href="public_estimator_api_estimators.md#tabforgeclassifier">Classifier · Regressor</a></p></div>
<a class="task-art" href="tutorials.md#prediction" aria-label="Prediction examples"><img src="assets/prediction.svg" alt="Tabular rows mapped to target predictions" width="240" height="120"></a>
<a class="task-examples" href="tutorials.md#prediction">Examples</a>
</section>
<section class="task-card">
<div class="task-card-body"><h3><a href="user_guide.md#embedding">Embedding</a></h3><p>Extract per-feature token embeddings or pool them into one embedding per row.</p><p><strong>Applications:</strong> Feed row embeddings into a classifier, retrieve similar records, or visualise cohorts with PCA.<br><strong>Estimator:</strong> <a href="public_estimator_api_estimators.md#tabforgeembedder">TabFORGEEmbedder</a></p></div>
<a class="task-art" href="tutorials.md#embedding" aria-label="Embedding examples"><img src="assets/embeddings.svg" alt="Layers of per-feature embedding grids" width="240" height="120"></a>
<a class="task-examples" href="tutorials.md#embedding">Examples</a>
</section>
<section class="task-card">
<div class="task-card-body"><h3><a href="user_guide.md#imputation">Imputation</a></h3><p>Fill missing or selected cells while retaining observed values.</p><p><strong>Applications:</strong> Complete unanswered survey fields or missing sensor measurements before analysis.<br><strong>Estimator:</strong> <a href="public_estimator_api_estimators.md#tabforgeimputer">TabFORGEImputer</a></p></div>
<a class="task-art" href="tutorials.md#imputation" aria-label="Imputation examples"><img src="assets/imputation.svg" alt="Missing table cells completed from observed context" width="240" height="120"></a>
<a class="task-examples" href="tutorials.md#imputation">Examples</a>
</section>
<section class="task-card">
<div class="task-card-body"><h3><a href="user_guide.md#clustering">Clustering</a></h3><p>Group similar rows using pooled tabular embeddings.</p><p><strong>Applications:</strong> Segment customers by purchasing patterns or group experiments with similar measurements.<br><strong>Estimator:</strong> <a href="public_estimator_api_estimators.md#tabforgeclusterer-and-tabforgeanomalydetector">TabFORGEClusterer</a></p></div>
<a class="task-art" href="tutorials.md#clustering" aria-label="Clustering examples"><img src="assets/clustering.svg" alt="Two distinct groups of row embeddings" width="240" height="120"></a>
<a class="task-examples" href="tutorials.md#clustering">Examples</a>
</section>
<section class="task-card">
<div class="task-card-body"><h3><a href="user_guide.md#anomaly-detection">Anomaly detection</a></h3><p>Score unusual rows in the pooled embedding space.</p><p><strong>Applications:</strong> Flag unusual equipment readings or transaction records for further inspection.<br><strong>Estimator:</strong> <a href="public_estimator_api_estimators.md#tabforgeclusterer-and-tabforgeanomalydetector">TabFORGEAnomalyDetector</a></p></div>
<a class="task-art" href="tutorials.md#anomaly-detection" aria-label="Anomaly detection examples"><img src="assets/anomaly.svg" alt="An outlier separated from a group of ordinary rows" width="240" height="120"></a>
<a class="task-examples" href="tutorials.md#anomaly-detection">Examples</a>
</section>
</div>

<section class="landing-container architecture-section">
<h2>Model architecture</h2>
<p>The frozen <strong>Structure-aware Feature Encoder</strong> supplies contextual feature tokens. The <strong>Score-based Diffusion Transformer</strong> learns their latent distribution, and the <strong>Denoising-aligned Latent Decoder</strong> reconstructs mixed-type table values.</p>
<figure class="paper-figure"><a href="assets/paper-framework.svg" target="_blank" rel="noopener"><img src="assets/paper-framework.svg" alt="Original paper architecture: mixed-type input dataset, Structure-aware Feature Encoder, Score-based Diffusion Transformer, and Denoising-aligned Latent Decoder" width="2400" height="592"></a></figure>
<p>For the two-stage design, sampling workflow, and citation, see the <a href="paper.md">paper</a> and <a href="latent_diffusion_model.md">diffusion and decoding reference</a>.</p>
</section>

<section class="home-more-info">
<div class="landing-container home-info-grid">
<section><h2>News</h2><p><strong>NeurIPS 2026.</strong> <em>TabFORGE: A Tabular Foundation Model for Generative Modelling</em> has been accepted at NeurIPS 2026. <a href="paper.md">Paper and citation</a>.</p><p><strong>Pretrained backbone:</strong> <a href="https://huggingface.co/XiangjianJiang/TabFORGE">Download model weights</a>.</p></section>
<section><h2>Documentation</h2><ul><li><a href="getting_started.md">Installation and first example</a></li><li><a href="user_guide.md">User Guide</a> · <a href="public_estimator_api.md">API Reference</a></li><li><a href="tutorials.md">Examples and Colab notebooks</a></li><li><a href="cli.md">Command-line workflows</a></li></ul></section>
<section><h2>Research &amp; development</h2><p>Browse the <a href="https://github.com/SilenceX12138/TabFORGE">source repository</a> and <a href="development.md">development guide</a>. TabFORGE-owned code uses Apache 2.0; vendored components retain their upstream terms.</p><a class="button orange" href="paper.md#citation">Cite TabFORGE</a></section>
</div>
</section>
