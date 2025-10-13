// Quiz data embedded directly
const QUIZ_DATA = {
    quiz1: `1) In which scenario is forward‑mode automatic differentiation the best fit?
☐ A. Computing sensitivities w.r.t. a few inputs for a function with many outputs
☐ B. Training a deep neural network with millions of parameters and a scalar loss
☐ C. Efficiently computing many VJPs (vector–Jacobian products)
☐ D. Finite differences for large parameter vectors
Show answer & explanation
2) In JAX, which transformation most directly vectorizes a function over a batch dimension without manual loops?
☐ A. pmap
☐ B. vmap
☐ C. jit
☐ D. grad
Show answer & explanation
3) On modern NVIDIA GPUs, which tensor layout often yields more coalesced memory access for many 2D conv workloads?
☐ A. NLC
☐ B. CHWN
☐ C. NHWC
☐ D. NCHW
Show answer & explanation
4) Regarding gradient checkpointing in reverse‑mode AD, which is most accurate?
☐ A. Cuts compute ~50% but increases memory
☐ B. Reduces memory O(log n) at ~10× compute
☐ C. Replaces backward with numerical differentiation
☐ D. Reduces peak memory (≈√n layers) with modest compute (~33%)
Show answer & explanation
5) For a ring all‑reduce with payload size P and N devices, the per‑device traffic per step (idealized) is:
☐ A. P·(N−1)/N
☐ B. 2·P/N
☐ C. 2·P·(N−1)/N
☐ D. P·(N−1)
Show answer & explanation
6) Which optimizer is most suitable for sparse updates (e.g., embeddings)?
☐ A. SGD with momentum
☐ B. Adam
☐ C. Adagrad
☐ D. RMSprop
Show answer & explanation
7) In PyTorch, which function is used to enable automatic mixed precision training?
☐ A. torch.cuda.amp.autocast
☐ B. torch.half
☐ C. torch.float16
☐ D. torch.optim.AdamW
Show answer & explanation
8) What is the primary advantage of using torch.nn.DataParallel over torch.nn.parallel.DistributedDataParallel?
☐ A. Better performance
☐ B. Easier to use (single-process, multi-threaded)
☐ C. Lower memory usage
☐ D. Better scalability
Show answer & explanation
9) Which of the following is NOT a valid attention mechanism in transformer models?
☐ A. Self-attention
☐ B. Cross-attention
☐ C. Multi-head attention
☐ D. Circular attention
Show answer & explanation
10) In the context of neural architecture search (NAS), what does "one-shot NAS" refer to?
☐ A. Searching for architecture in a single training run
☐ B. Using a single evaluation metric
☐ C. Testing only one architecture
☐ D. Using a single GPU
Show answer & explanation
11) Which activation function is generally recommended for hidden layers in deep networks?
☐ A. Sigmoid
☐ B. Tanh
☐ C. ReLU
☐ D. Leaky ReLU
Show answer & explanation
12) What is the purpose of batch normalization?
☐ A. To normalize inputs between 0 and 1
☐ B. To reduce internal covariate shift
☐ C. To regularize the network
☐ D. To speed up convergence
Show answer & explanation
13) In transfer learning, what is "fine-tuning"?
☐ A. Training from scratch
☐ B. Adjusting pre-trained weights on a new dataset
☐ C. Using pre-trained features as fixed features
☐ D. Data augmentation
Show answer & explanation
14) Which technique is used to prevent overfitting in neural networks?
☐ A. Increasing model size
☐ B. Dropout
☐ C. Removing regularization
☐ D. Using smaller datasets
Show answer & explanation
15) What is the purpose of learning rate scheduling?
☐ A. To keep learning rate constant
☐ B. To adjust learning rate during training
☐ C. To increase learning rate over time
☐ D. To disable learning rate entirely
Show answer & explanation
16) Which loss function is most appropriate for multi-class classification?
☐ A. Mean Squared Error
☐ B. Binary Cross-Entropy
☐ C. Categorical Cross-Entropy
☐ D. Hinge Loss
Show answer & explanation
17) What is the main advantage of using residual connections (ResNets)?
☐ A. Reduced computational cost
☐ B. Easier optimization of deep networks
☐ C. Better generalization
☐ D. Faster inference
Show answer & explanation
18) In the context of GANs, what is the role of the discriminator?
☐ A. To generate fake data
☐ B. To distinguish real from fake data
☐ C. To optimize the generator
☐ D. To regularize training
Show answer & explanation
19) Which normalization technique is most effective for Recurrent Neural Networks?
☐ A. Batch Normalization
☐ B. Layer Normalization
☐ C. Instance Normalization
☐ D. Group Normalization
Show answer & explanation
20) What is the purpose of early stopping in neural network training?
☐ A. To speed up training
☐ B. To prevent overfitting
☐ C. To improve accuracy
☐ D. To reduce memory usage
Show answer & explanation
21) Which of the following is a characteristic of unsupervised learning?
☐ A. Requires labeled data
☐ B. Learns patterns from unlabeled data
☐ C. Always better than supervised learning
☐ D. Used only for classification
Show answer & explanation
22) In reinforcement learning, what is "exploration vs exploitation"?
☐ A. Choosing between different algorithms
☐ B. Balancing trying new actions vs using known good actions
☐ C. Selecting hyperparameters
☐ D. Optimizing rewards only
Show answer & explanation
23) Which metric is most appropriate for evaluating imbalanced classification?
☐ A. Accuracy
☐ B. Precision
☐ C. F1-Score
☐ D. All of the above
Show answer & explanation
24) What is the curse of dimensionality?
☐ A. Training becomes faster with more dimensions
☐ B. Data becomes sparse in high-dimensional spaces
☐ C. Models always perform better
☐ D. Memory requirements decrease
Show answer & explanation
25) Which technique is used for dimensionality reduction?
☐ A. Random Forest
☐ B. PCA (Principal Component Analysis)
☐ C. Decision Tree
☐ D. Support Vector Machine
Show answer & explanation
26) In ensemble learning, what is "bagging"?
☐ A. Training models sequentially
☐ B. Training models in parallel on bootstrap samples
☐ C. Combining different model types
☐ D. Optimizing a single model
Show answer & explanation
27) What is the main purpose of cross-validation?
☐ A. To train faster
☐ B. To evaluate model generalization
☐ C. To reduce training data
☐ D. To increase model complexity
Show answer & explanation
28) Which algorithm is best suited for large-scale linear classification?
☐ A. K-Nearest Neighbors
☐ B. Decision Trees
☐ C. Linear SVM
☐ D. Neural Networks
Show answer & explanation
29) What is overfitting?
☐ A. Model performs well on training data but poorly on test data
☐ B. Model performs poorly on both training and test data
☐ C. Model performs equally well on both
☐ D. Model is too simple
Show answer & explanation
30) Which regularization technique adds a penalty based on the absolute values of weights?
☐ A. L2 Regularization
☐ B. L1 Regularization
☐ C. Dropout
☐ D. Early Stopping
Show answer & explanation
31) In the context of neural networks, what is "weight decay"?
☐ A. Weights becoming zero over time
☐ B. L2 regularization
☐ C. Learning rate decay
☐ D. Network pruning
Show answer & explanation
32) Which optimizer is an extension of SGD with adaptive learning rates?
☐ A. Momentum
☐ B. Adam
☐ C. RMSprop
☐ D. All of the above
Show answer & explanation
33) What is the purpose of data augmentation?
☐ A. To reduce dataset size
☐ B. To artificially increase dataset diversity
☐ C. To speed up training
☐ D. To clean data
Show answer & explanation
34) Which activation function is zero-centered and helps with the vanishing gradient problem?
☐ A. Sigmoid
☐ B. Tanh
☐ C. ReLU
☐ D. Softmax
Show answer & explanation
35) What is the main advantage of using convolutional neural networks for image data?
☐ A. Fewer parameters than fully connected networks
☐ B. Better performance on text data
☐ C. Simpler architecture
☐ D. Faster training
Show answer & explanation
36) In the context of NLP, what is "word embedding"?
☐ A. Converting words to numerical vectors
☐ B. Translating text
☐ C. Tokenizing sentences
☐ D. Text preprocessing
Show answer & explanation
37) Which architecture is specifically designed for sequence-to-sequence tasks?
☐ A. CNN
☐ B. RNN
☐ C. Transformer
☐ D. Autoencoder
Show answer & explanation
38) What is "attention mechanism" in neural networks?
☐ A. A way to focus on important parts of input
☐ B. A regularization technique
☐ C. A normalization method
☐ D. An optimization algorithm
Show answer & explanation
39) Which of the following is a characteristic of deep learning?
☐ A. Always requires large datasets
☐ B. Learns hierarchical representations
☐ C. Only works for images
☐ D. Cannot be interpreted
Show answer & explanation
40) What is the purpose of "backpropagation"?
☐ A. To forward pass data through network
☐ B. To compute gradients for weight updates
☐ C. To initialize weights
☐ D. To regularize the model
Show answer & explanation

Answers & Explanations
1) A — Forward-mode AD computes directional derivatives efficiently for many outputs w.r.t. few inputs, making it ideal for sensitivity analysis.
2) B — vmap (vectorizing map) automatically vectorizes functions over batch dimensions, eliminating manual loops.
3) C — NHWC (Batch, Height, Width, Channels) layout generally provides better memory coalescing on NVIDIA GPUs.
4) D — Gradient checkpointing reduces peak memory to roughly √n layers with about 33% additional computation.
5) C — Ring all-reduce requires each device to send and receive 2·P·(N−1)/N data per step in the ideal case.
6) C — Adagrad is particularly well-suited for sparse updates due to its per-parameter learning rate adaptation.
7) A — torch.cuda.amp.autocast enables automatic mixed precision training in PyTorch.
8) B — DataParallel is easier to use as it operates in a single process with multiple threads.
9) D — Circular attention is not a standard attention mechanism in transformer models.
10) A — One-shot NAS searches for architectures in a single training run, typically using a weight-sharing approach.
11) C — ReLU is generally recommended for hidden layers due to its simplicity and ability to mitigate vanishing gradients.
12) B — Batch normalization reduces internal covariate shift by normalizing layer inputs.
13) B — Fine-tuning involves adjusting pre-trained weights on a new dataset while leveraging learned features.
14) B — Dropout randomly deactivates neurons during training to prevent overfitting.
15) B — Learning rate scheduling adjusts the learning rate during training to improve convergence.
16) C — Categorical cross-entropy is the standard loss function for multi-class classification problems.
17) B — Residual connections make it easier to optimize very deep networks by providing gradient pathways.
18) B — The discriminator's role is to distinguish between real and fake data in GAN training.
19) B — Layer normalization is most effective for RNNs as it normalizes across features rather than batches.
20) B — Early stopping prevents overfitting by stopping training when validation performance degrades.
21) B — Unsupervised learning learns patterns from unlabeled data without explicit labels.
22) B — Exploration vs exploitation balances trying new actions versus using known good actions in RL.
23) D — All metrics (accuracy, precision, F1-score) should be considered for imbalanced classification.
24) B — The curse of dimensionality refers to data becoming sparse in high-dimensional spaces.
25) B — PCA is a widely used technique for dimensionality reduction.
26) B — Bagging trains models in parallel on bootstrap samples of the training data.
27) B — Cross-validation evaluates model generalization by using different train-test splits.
28) C — Linear SVM is efficient and effective for large-scale linear classification.
29) A — Overfitting occurs when a model performs well on training data but poorly on test data.
30) B — L1 regularization adds a penalty based on the absolute values of weights, promoting sparsity.
31) B — Weight decay is another term for L2 regularization in neural networks.
32) D — Adam, RMSprop, and Momentum are all extensions of SGD with adaptive learning rates.
33) B — Data augmentation artificially increases dataset diversity through transformations.
34) B — Tanh is zero-centered and helps mitigate vanishing gradient problems compared to sigmoid.
35) A — CNNs have fewer parameters than fully connected networks due to weight sharing.
36) A — Word embedding converts words to dense numerical vectors that capture semantic relationships.
37) C — Transformers are specifically designed for sequence-to-sequence tasks with self-attention.
38) A — Attention mechanisms allow models to focus on important parts of the input sequence.
39) B — Deep learning learns hierarchical representations automatically from data.
40) B — Backpropagation computes gradients for weight updates using the chain rule.`,

    quiz2: `1) Which of the following are key advantages of Federated Learning? (Select all that apply)
☐ A. Improved privacy by keeping data on local devices
☐ B. Reduced communication overhead compared to centralized training
☐ C. Better utilization of distributed computational resources
☐ D. Guaranteed model convergence
Show answer & explanation
2) In the context of model quantization, which statement is most accurate?
☐ A. Quantization always improves model accuracy
☐ B. Post-training quantization requires retraining the model
☐ C. Quantization-aware training typically yields better accuracy than post-training quantization
☐ D. 8-bit quantization is the only supported format
Show answer & explanation
3) Which optimization technique is most suitable for deploying models on resource-constrained edge devices?
☐ A. Increasing model depth
☐ B. Knowledge distillation
☐ C. Using larger batch sizes
☐ D. Adding more regularization
Show answer & explanation
4) What is the primary challenge in implementing continual learning?
☐ A. Catastrophic forgetting
☐ B. Overfitting
☐ C. Vanishing gradients
☐ D. Exploding gradients
Show answer & explanation
5) Which of the following are benefits of Neural Architecture Search (NAS)? (Select all that apply)
☐ A. Automated architecture design
☐ B. Discovery of novel network topologies
☐ C. Reduced human expertise requirements
☐ D. Guaranteed optimal architecture
Show answer & explanation
6) In transfer learning, what is the main difference between feature extraction and fine-tuning?
☐ A. Feature extraction updates all layers, fine-tuning updates only the final layer
☐ B. Feature extraction freezes pre-trained weights, fine-tuning updates some or all weights
☐ C. Feature extraction requires more data than fine-tuning
☐ D. Fine-tuning is always faster than feature extraction
Show answer & explanation
7) Which technique is most effective for handling class imbalance in deep learning?
☐ A. Using larger batch sizes
☐ B. Focal loss
☐ C. Removing minority class samples
☐ D. Using only accuracy as metric
Show answer & explanation
8) What is the main purpose of model pruning?
☐ A. To increase model size
☐ B. To improve training speed
☐ C. To reduce model size and inference time
☐ D. To increase model accuracy
Show answer & explanation
9) Which of the following are common approaches to few-shot learning? (Select all that apply)
☐ A. Meta-learning
☐ B. Data augmentation
☐ C. Transfer learning
☐ D. Using very large datasets
Show answer & explanation
10) In self-supervised learning, what is a "pretext task"?
☐ A. The final downstream task
☐ B. An auxiliary task used to learn representations
☐ C. Data preprocessing step
☐ D. Model evaluation task
Show answer & explanation
11) Which regularization technique is most effective for preventing overfitting in small datasets?
☐ A. Data augmentation
☐ B. Increasing model size
☐ C. Reducing regularization strength
☐ D. Using larger learning rates
Show answer & explanation
12) What is the main advantage of using mixed precision training?
☐ A. Better model accuracy
☐ B. Reduced memory usage and faster training
☐ C. Simpler implementation
☐ D. Guaranteed convergence
Show answer & explanation
13) Which optimization algorithm is most suitable for training very large language models?
☐ A. Basic SGD
☐ B. AdamW
☐ C. RMSprop
☐ D. Adagrad
Show answer & explanation
14) In the context of model compression, what does "weight sharing" refer to?
☐ A. Using the same weights across different layers
☐ B. Sharing weights between multiple models
☐ C. Grouping similar weights and using a single representative value
☐ D. Distributing weights across multiple devices
Show answer & explanation
15) Which of the following are key considerations for deploying models on mobile devices? (Select all that apply)
☐ A. Model size constraints
☐ B. Power consumption
☐ C. Inference latency requirements
☐ D. Unlimited memory availability
Show answer & explanation
16) What is the primary goal of curriculum learning?
☐ A. To train on random samples
☐ B. To gradually increase task difficulty during training
☐ C. To use only the hardest examples
☐ D. To ignore easy examples completely
Show answer & explanation
17) Which technique is most effective for domain adaptation?
☐ A. Training only on source domain data
☐ B. Using domain-adversarial training
☐ C. Ignoring domain differences
☐ D. Using the same architecture for all domains
Show answer & explanation
18) In the context of efficient transformers, which method reduces computational complexity?
☐ A. Using full attention mechanism
☐ B. Sparse attention patterns
☐ C. Increasing sequence length
☐ D. Adding more attention heads
Show answer & explanation
19) Which of the following are benefits of using ensemble methods? (Select all that apply)
☐ A. Improved robustness
☐ B. Better generalization
☐ C. Reduced overfitting
☐ D. Always better computational efficiency
Show answer & explanation
20) What is the main challenge in lifelong learning systems?
☐ A. Lack of computational resources
☐ B. Balancing new learning with knowledge retention
☐ C. Too much available data
☐ D. Simple implementation requirements
Show answer & explanation
21) Which technique is most suitable for handling long-term dependencies in sequences?
☐ A. Simple RNN
☐ B. LSTM
☐ C. CNN
☐ D. Perceptron
Show answer & explanation
22) In neural architecture search, what does "differentiable architecture search" refer to?
☐ A. Using discrete search methods
☐ B. Optimizing architecture as a continuous relaxation
☐ C. Random architecture selection
☐ D. Manual architecture design
Show answer & explanation
23) Which of the following are common challenges in multi-task learning? (Select all that apply)
☐ A. Task interference
☐ B. Different task difficulties
☐ C. Unequal data distribution across tasks
☐ D. All tasks always perform equally well
Show answer & explanation
24) What is the main purpose of using "early stopping" in training?
☐ A. To always train for maximum epochs
☐ B. To prevent overfitting by stopping when validation performance plateaus
☐ C. To speed up initial training
☐ D. To use only training data for evaluation
Show answer & explanation
25) Which optimization technique specifically addresses the vanishing gradient problem?
☐ A. Using sigmoid activations only
☐ B. Batch normalization
☐ C. Removing residual connections
☐ D. Using deeper networks only
Show answer & explanation
26) In federated learning, what is "client selection"?
☐ A. Selecting the best performing clients only
☐ B. Choosing which clients participate in each training round
☐ C. Selecting clients with most data
☐ D. Using all clients in every round
Show answer & explanation
27) Which of the following are advantages of knowledge distillation? (Select all that apply)
☐ A. Model compression
☐ B. Improved generalization
☐ C. Reduced inference time
☐ D. Always requires teacher model to be larger
Show answer & explanation
28) What is the main challenge in implementing models on embedded systems?
☐ A. Unlimited computational resources
☐ B. Memory and power constraints
☐ C. Large storage availability
☐ D. No latency requirements
Show answer & explanation
29) Which technique is most effective for handling out-of-distribution detection?
☐ A. Training only on in-distribution data
☐ B. Using maximum softmax probability
☐ C. Ignoring uncertainty estimates
☐ D. Using only accuracy metrics
Show answer & explanation
30) In the context of model optimization, what does "neural architecture search" aim to achieve?
☐ A. Find optimal network architecture automatically
☐ B. Manually design better networks
☐ C. Use only existing architectures
☐ D. Focus on hyperparameter tuning only
Show answer & explanation
31) Which of the following are key considerations for real-time inference? (Select all that apply)
☐ A. Low latency requirements
☐ B. Model complexity
☐ C. Hardware constraints
☐ D. Unlimited computational budget
Show answer & explanation
32) What is the main advantage of using "transformers" over RNNs for sequence tasks?
☐ A. Always smaller model size
☐ B. Better handling of long-range dependencies
☐ C. Simpler architecture
☐ D. Always faster training
Show answer & explanation
33) Which technique is most suitable for reducing model size while maintaining performance?
☐ A. Adding more layers
☐ B. Pruning and quantization
☐ C. Using larger batch sizes
☐ D. Removing all regularization
Show answer & explanation
34) In self-supervised learning, what is "contrastive learning"?
☐ A. Learning without any objective function
☐ B. Learning by comparing similar and dissimilar samples
☐ C. Using only labeled data
☐ D. Ignoring data relationships
Show answer & explanation
35) Which of the following are benefits of using "efficient attention" mechanisms? (Select all that apply)
☐ A. Reduced computational complexity
☐ B. Better handling of long sequences
☐ C. Lower memory usage
☐ D. Always better accuracy
Show answer & explanation
36) What is the main challenge in "multi-modal learning"?
☐ A. Dealing with only one data type
☐ B. Aligning representations across different modalities
☐ C. Using simple architectures only
☐ D. No data preprocessing needed
Show answer & explanation
37) Which technique is most effective for "continual learning"?
☐ A. Training on all data at once
☐ B. Using regularization to prevent catastrophic forgetting
☐ C. Ignoring new data
☐ D. Resetting model weights
Show answer & explanation
38) In the context of "edge AI", what is the primary constraint?
☐ A. Unlimited computational resources
☐ B. Power and memory limitations
☐ C. No latency constraints
☐ D. Large storage availability
Show answer & explanation
39) Which of the following are important aspects of "responsible AI"? (Select all that apply)
☐ A. Fairness and bias mitigation
☐ B. Model interpretability
☐ C. Privacy preservation
☐ D. Ignoring ethical considerations
Show answer & explanation
40) What is the main advantage of using "transfer learning" in computer vision?
☐ A. Always better than training from scratch
☐ B. Leveraging pre-trained features to reduce data requirements
☐ C. No need for domain adaptation
☐ D. Simpler architectures only
Show answer & explanation

Answers & Explanations
1) A, C — Federated learning improves privacy by keeping data local and better utilizes distributed resources, though it may increase communication overhead and doesn't guarantee convergence.
2) C — Quantization-aware training typically yields better accuracy than post-training quantization as it accounts for quantization effects during training.
3) B — Knowledge distillation trains a smaller "student" model to mimic a larger "teacher" model, effectively reducing model size for edge deployment.
4) A — Catastrophic forgetting is the primary challenge in continual learning, where new learning interferes with previously acquired knowledge.
5) A, B, C — NAS enables automated architecture design, discovery of novel topologies, and reduced human expertise requirements, but doesn't guarantee optimal architectures.
6) B — Feature extraction freezes pre-trained weights while fine-tuning updates some or all weights during training on the new task.
7) B — Focal loss addresses class imbalance by focusing on hard-to-classify examples and down-weighting easy examples.
8) C — Model pruning reduces model size and inference time by removing unnecessary weights or connections.
9) A, B, C — Meta-learning, data augmentation, and transfer learning are common approaches to few-shot learning; using very large datasets is not.
10) B — A pretext task is an auxiliary task designed to learn useful representations without requiring manual labels.
11) A — Data augmentation is most effective for preventing overfitting in small datasets by artificially increasing training data diversity.
12) B — Mixed precision training reduces memory usage and speeds up training by using lower precision calculations.
13) B — AdamW is optimized for training very large language models, incorporating weight decay regularization.
14) C — Weight sharing groups similar weights and uses representative values, reducing model size and complexity.
15) A, B, C — Mobile deployment requires considering model size constraints, power consumption, and inference latency; memory is limited, not unlimited.
16) B — Curriculum learning gradually increases task difficulty during training, starting with easy examples and progressing to harder ones.
17) B — Domain-adversarial training effectively aligns feature distributions across different domains for domain adaptation.
18) B — Sparse attention patterns reduce computational complexity from O(n²) to O(n log n) or better for long sequences.
19) A, B, C — Ensemble methods improve robustness, generalization, and can reduce overfitting, but may not be computationally efficient.
20) B — The main challenge in lifelong learning is balancing new learning with knowledge retention to avoid catastrophic forgetting.
21) B — LSTMs are specifically designed to handle long-term dependencies in sequences through gating mechanisms.
22) B — Differentiable architecture search optimizes architecture as a continuous relaxation, allowing gradient-based optimization.
23) A, B, C — Multi-task learning faces challenges with task interference, varying difficulties, and unequal data distributions.
24) B — Early stopping prevents overfitting by monitoring validation performance and stopping when it stops improving.
25) B — Batch normalization addresses vanishing gradients by normalizing layer inputs and maintaining stable gradients.
26) B — Client selection chooses which devices participate in each federated learning round based on various criteria.
27) A, B, C — Knowledge distillation enables model compression, can improve generalization, and reduces inference time.
28) B — Embedded systems face significant memory and power constraints that limit model complexity and size.
29) B — Maximum softmax probability is an effective method for out-of-distribution detection by measuring model uncertainty.
30) A — Neural architecture search aims to find optimal network architectures automatically rather than relying on manual design.
31) A, B, C — Real-time inference requires considering low latency, model complexity, and hardware constraints; computational budget is limited.
32) B — Transformers better handle long-range dependencies through self-attention mechanisms compared to sequential RNN processing.
33) B — Pruning and quantization effectively reduce model size while maintaining acceptable performance levels.
34) B — Contrastive learning learns representations by comparing similar and dissimilar sample pairs in the feature space.
35) A, B, C — Efficient attention mechanisms reduce computational complexity, handle long sequences better, and use less memory, though accuracy may vary.
36) B — Multi-modal learning's main challenge is aligning representations across different data types (images, text, audio, etc.).
37) B — Continual learning uses regularization techniques to prevent catastrophic forgetting when learning new tasks sequentially.
38) B — Edge AI primarily faces power and memory limitations that constrain model size and computational complexity.
39) A, B, C — Responsible AI involves fairness, bias mitigation, model interpretability, and privacy preservation.
40) B — Transfer learning leverages pre-trained features to reduce data requirements and training time for new tasks.`
};

// Quiz application state
let currentQuiz = null;
let currentQuestionIndex = 0;
let userAnswers = [];
let parsedQuizzes = {};

// Parse quiz data
function parseQuizData() {
    for (const quizName in QUIZ_DATA) {
        const content = QUIZ_DATA[quizName];
        const questions = [];
        const lines = content.split('\n');
        let currentQuestion = null;
        let inAnswersSection = false;

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();

            // Check for answers section
            if (line === 'Answers & Explanations') {
                inAnswersSection = true;
                continue;
            }

            // Parse questions
            if (line.match(/^\d+\)/) && !inAnswersSection) {
                // Save previous question
                if (currentQuestion) {
                    questions.push(currentQuestion);
                }

                // Start new question
                currentQuestion = {
                    number: parseInt(line.match(/^\d+/)[0]),
                    question: line.substring(line.indexOf(')') + 1).trim(),
                    options: [],
                    type: 'single'
                };

                // Read options
                while (i + 1 < lines.length && lines[i + 1].trim().match(/^☐/)) {
                    i++;
                    const optionLine = lines[i].trim();
                    const optionMatch = optionLine.match(/^☐ (.+)$/);
                    if (optionMatch) {
                        currentQuestion.options.push({
                            text: optionMatch[1],
                            letter: String.fromCharCode(65 + currentQuestion.options.length)
                        });
                    }
                }
            }

            // Check for multiple choice
            if (line.match(/\(Select all that apply\)/)) {
                if (currentQuestion) {
                    currentQuestion.type = 'multiple';
                }
            }

            // Parse answers
            if (inAnswersSection && line.match(/^\d+\)/)) {
                const answerMatch = line.match(/^\d+\)\s*(.+?)\s*—\s*(.+?)\s*(?:Back|$)/);
                if (answerMatch) {
                    const questionNum = parseInt(line.match(/^\d+/)[0]);
                    const answer = answerMatch[1].trim();
                    const explanation = answerMatch[2].trim();

                    const question = questions.find(q => q.number === questionNum);
                    if (question) {
                        question.correctAnswer = answer;
                        question.explanation = explanation;
                        question.chineseExplanation = explanation; // Using same explanation for now
                    }
                }
            }
        }

        // Add last question
        if (currentQuestion) {
            questions.push(currentQuestion);
        }

        parsedQuizzes[quizName] = questions;
    }
}

// Page navigation
function showPage(pageId) {
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    document.getElementById(pageId).classList.add('active');
}

// Start quiz
function startQuiz(quizName) {
    currentQuiz = quizName;
    currentQuestionIndex = 0;
    userAnswers = new Array(parsedQuizzes[quizName].length).fill(null);

    showPage('quiz-page');
    showQuestion(0);
    updateProgress();
}

// Show question
function showQuestion(index) {
    const question = parsedQuizzes[currentQuiz][index];
    if (!question) return;

    // Update question info
    document.getElementById('question-number').textContent = `第 ${question.number} 题`;
    document.getElementById('question-type').textContent = question.type === 'multiple' ? '多选题' : '单选题';
    document.getElementById('question-text').textContent = question.question;

    // Generate options
    const optionsContainer = document.getElementById('options-container');
    optionsContainer.innerHTML = '';

    question.options.forEach((option, optionIndex) => {
        const optionDiv = document.createElement('div');
        optionDiv.className = 'option';

        const currentValue = userAnswers[index];
        const isChecked = Array.isArray(currentValue)
            ? currentValue.includes(option.letter)
            : currentValue === option.letter;

        if (question.type === 'multiple') {
            optionDiv.innerHTML = `
                <label class="option-label">
                    <input type="checkbox" name="option" value="${option.letter}" ${isChecked ? 'checked' : ''}>
                    <span class="option-letter">${option.letter}</span>
                    <span class="option-text">${option.text}</span>
                </label>
            `;
        } else {
            optionDiv.innerHTML = `
                <label class="option-label">
                    <input type="radio" name="option" value="${option.letter}" ${isChecked ? 'checked' : ''}>
                    <span class="option-letter">${option.letter}</span>
                    <span class="option-text">${option.text}</span>
                </label>
            `;
        }

        optionsContainer.appendChild(optionDiv);
    });

    // Update navigation buttons
    updateNavigationButtons();
    updateSubmitButton();
}

// Handle answer selection
function handleAnswerSelection() {
    const question = parsedQuizzes[currentQuiz][currentQuestionIndex];
    let selectedAnswers = [];

    if (question.type === 'multiple') {
        const checkboxes = document.querySelectorAll('input[type="checkbox"]:checked');
        selectedAnswers = Array.from(checkboxes).map(cb => cb.value);
    } else {
        const radio = document.querySelector('input[type="radio"]:checked');
        selectedAnswers = radio ? radio.value : '';
    }

    userAnswers[currentQuestionIndex] = selectedAnswers;
    updateSubmitButton();
}

// Update navigation buttons
function updateNavigationButtons() {
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');

    const isFirstQuestion = currentQuestionIndex === 0;
    const isLastQuestion = currentQuestionIndex === parsedQuizzes[currentQuiz].length - 1;

    prevBtn.disabled = isFirstQuestion;
    nextBtn.textContent = isLastQuestion ? '完成' : '下一题';
}

// Update submit button
function updateSubmitButton() {
    const submitBtn = document.getElementById('submit-btn');
    const currentAnswer = userAnswers[currentQuestionIndex];
    const hasAnswer = currentAnswer && (Array.isArray(currentAnswer) ? currentAnswer.length > 0 : currentAnswer !== '');

    submitBtn.disabled = !hasAnswer;
}

// Update progress
function updateProgress() {
    const progress = ((currentQuestionIndex + 1) / parsedQuizzes[currentQuiz].length) * 100;
    document.getElementById('progress-fill').style.width = `${progress}%`;
    document.getElementById('progress-text').textContent = `第 ${currentQuestionIndex + 1} 题 / ${parsedQuizzes[currentQuiz].length} 题`;
}

// Navigation functions
function previousQuestion() {
    if (currentQuestionIndex > 0) {
        currentQuestionIndex--;
        showQuestion(currentQuestionIndex);
        updateProgress();
    }
}

function nextQuestion() {
    if (currentQuestionIndex < parsedQuizzes[currentQuiz].length - 1) {
        currentQuestionIndex++;
        showQuestion(currentQuestionIndex);
        updateProgress();
    } else {
        completeQuiz();
    }
}

function submitAnswer() {
    const question = parsedQuizzes[currentQuiz][currentQuestionIndex];
    const userAnswer = userAnswers[currentQuestionIndex];

    if (!userAnswer || (Array.isArray(userAnswer) && userAnswer.length === 0)) {
        return;
    }

    const isCorrect = checkAnswer(question, userAnswer);

    if (!isCorrect) {
        showExplanation(question, userAnswer);
    } else {
        setTimeout(() => {
            nextQuestion();
        }, 500);
    }
}

// Check answer
function checkAnswer(question, userAnswer) {
    if (question.type === 'multiple') {
        const correctAnswers = question.correctAnswer.split(', ').map(a => a.trim());
        const userAnswersArray = Array.isArray(userAnswer) ? userAnswer : [userAnswer];

        if (correctAnswers.length !== userAnswersArray.length) {
            return false;
        }

        return correctAnswers.every(ans => userAnswersArray.includes(ans.split(' ')[0]));
    } else {
        return question.correctAnswer.startsWith(userAnswer);
    }
}

// Show explanation
function showExplanation(question, userAnswer) {
    const explanationContainer = document.getElementById('explanation-container');

    document.getElementById('correct-answer-text').textContent = question.correctAnswer;
    document.getElementById('explanation-text-content').textContent = question.explanation;
    document.getElementById('chinese-explanation-content').textContent = question.chineseExplanation;

    explanationContainer.style.display = 'block';

    // Highlight answers
    const options = document.querySelectorAll('.option');
    options.forEach(option => {
        const input = option.querySelector('input');

        // Highlight user's answer
        if ((Array.isArray(userAnswer) && userAnswer.includes(input.value)) || userAnswer === input.value) {
            option.classList.add('incorrect');
        }

        // Highlight correct answer
        if (question.type === 'multiple') {
            const correctAnswers = question.correctAnswer.split(', ').map(a => a.trim()[0]);
            if (correctAnswers.includes(input.value)) {
                option.classList.add('correct');
            }
        } else {
            if (question.correctAnswer.startsWith(input.value)) {
                option.classList.add('correct');
            }
        }

        input.disabled = true;
    });
}

// Close explanation
function closeExplanation() {
    const explanationContainer = document.getElementById('explanation-container');
    explanationContainer.style.display = 'none';

    // Re-enable options
    const options = document.querySelectorAll('input');
    options.forEach(option => {
        option.disabled = false;
    });

    // Remove highlights
    document.querySelectorAll('.option').forEach(option => {
        option.classList.remove('correct', 'incorrect');
    });
}

// Continue to next question
function continueToNext() {
    closeExplanation();
    nextQuestion();
}

// Complete quiz
function completeQuiz() {
    let correctCount = 0;
    parsedQuizzes[currentQuiz].forEach((question, index) => {
        if (checkAnswer(question, userAnswers[index])) {
            correctCount++;
        }
    });

    const score = Math.round((correctCount / parsedQuizzes[currentQuiz].length) * 100);

    document.getElementById('completion-container').style.display = 'block';
    document.getElementById('final-score').textContent = `你的得分：${score} 分 (${correctCount}/${parsedQuizzes[currentQuiz].length})`;
    document.getElementById('score-breakdown').textContent = getScoreBreakdown(correctCount);
}

// Get score breakdown
function getScoreBreakdown(correctCount) {
    const total = parsedQuizzes[currentQuiz].length;
    const percentage = Math.round((correctCount / total) * 100);

    if (percentage >= 90) return '优秀！掌握得很好！';
    if (percentage >= 80) return '良好，继续保持！';
    if (percentage >= 70) return '中等，需要加强练习！';
    if (percentage >= 60) return '及格，还需努力！';
    return '不及格，需要认真复习！';
}

// Restart quiz
function restartQuiz() {
    currentQuestionIndex = 0;
    userAnswers = new Array(parsedQuizzes[currentQuiz].length).fill(null);

    document.getElementById('completion-container').style.display = 'none';
    showQuestion(0);
    updateProgress();
}

// Review answers
function reviewAnswers() {
    alert('错题复习功能开发中...');
}

// Go home
function goHome() {
    showPage('home-page');
    document.getElementById('completion-container').style.display = 'none';
    document.getElementById('explanation-container').style.display = 'none';
}

// Setup event listeners
function setupEventListeners() {
    const optionsContainer = document.getElementById('options-container');
    if (optionsContainer) {
        optionsContainer.addEventListener('change', handleAnswerSelection);
    }
}

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    parseQuizData();
    setupEventListeners();
    showPage('home-page');
});