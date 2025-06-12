weebit_path = "Dataset/weebit_reextracted.tsv"
onestopenglish_path = "iastate/onestop_english"
model = "deepseek-r1:14b" #(options: deepseek-r1:14b,deepseek-r1:32b,deepseek, qwen, llama)
corpus = 'weebit' #(options: weebit, onestopenglish)
level = 5
acc_output_path = "level.csv"

zeroShot_prompt = """
        Analyze the readability of the following text step by step.

        Text: {text}

        Let's think step by step:

        1. Vocabulary Complexity Analysis:
           - Identify difficult or uncommon words
           - Assess technical terminology usage
           - Evaluate word length distribution

        2. Sentence Structure Analysis:
           - Examine sentence length variation
           - Identify complex grammatical structures
           - Check for nested clauses and subordination

        3. Cohesion and Text Organization:
           - Evaluate logical flow between sentences
           - Check transition words and connectives
           - Assess paragraph structure

        4. Background Knowledge Requirements:
           - Identify assumed prior knowledge
           - Check for unexplained concepts
           - Evaluate cultural references

        Based on this analysis, determine the readability level:
        [Easy/Elementary] - [Medium/Intermediate] - [Hard/Advanced]

        Final Assessment:
        """

newPrompt = """
Analyze the readability of the following text using evidence-based Chain of Thought reasoning.

Text: {text}

Let's think step by step, analyzing each sentence individually before aggregating:

1. **Sentence-Level Vocabulary Complexity Analysis** (Primary Factor - 40% weight):
   Research Foundation: Lorge (1939) established "vocabulary load is the most important concomitant of difficulty"
   
   For each sentence:
   - Identify rare words (frequency <10 per million in COCA corpus)
   - Calculate percentage of uncommon vocabulary per sentence
   - Assess multisyllabic and morphologically complex words
   - Evaluate abstract vs concrete concepts
   - Track vocabulary difficulty progression across sentences
   - Note: Research shows 95% vs 66% accuracy improvement with stratified word familiarity analysis

2. **Sentence-Level Syntactic Structure Analysis** (Enhanced Factor - 25% weight):
   Research Foundation: Psycholinguistic measures significantly improve performance over baseline metrics
   
   For each sentence:
   - Count embedded clauses and subordination depth (TAASSC methodology)
   - Measure nominal subjects per clause and phrasal complexity
   - Identify passive voice, nominalizations, and non-finite constructions
   - Assess sentence length and structural variety
   - Evaluate coordination vs subordination patterns
   - Note: Parse-tree syntactic features show measurable processing time correlation

3. **Inter-Sentence Cohesion and Coherence Analysis** (Critical Factor - 20% weight):
   Research Foundation: Kintsch studies show "central role of coherence in reading ease"
   
   Between sentences:
   - Track entity references and pronoun chains (anaphoric distance)
   - Analyze lexical overlap between consecutive sentences
   - Evaluate topic progression and thematic consistency
   - Identify discourse markers and transition signals
   - Assess logical flow and argument structure
   - Note: Coh-Metrix coherence indices outperform traditional readability formulas

4. **Sentence-Level Background Knowledge Assessment** (Reader-Focused - 10% weight):
   Research Foundation: Reader background knowledge integration affects comprehension
   
   For each sentence:
   - Identify assumed prior knowledge domains
   - Assess cultural and contextual assumptions
   - Evaluate prerequisite concept density
   - Analyze inference requirements between sentences
   - Consider working memory load from sentence complexity
   - Note: Personalized assessment shows 38.3% improvement over traditional formulas

5. **Integrated Sentence-to-Text Analysis** (Synthesis - 5% weight):
   Research Foundation: Multi-component analysis consistently outperforms single-metric approaches
   
   - Assess vocabulary-syntax interactions within sentences
   - Identify bottleneck sentences creating difficulty spikes
   - Evaluate coherence compensation effects
   - Track cumulative cognitive load progression
   - Consider target audience matching per sentence

**Accuracy Validation Checks:**
- Sentence consistency: Do all factors align within each sentence?
- Inter-sentence coherence: Are assessments logically connected?
- Outlier detection: Any sentences dramatically inconsistent?
- Explainability: Can each sentence's difficulty be justified?

**Evidence-Based Difficulty Weighting:**
Using research-validated factor importance:
- Vocabulary complexity: 40% (Lorge 1939, TAALES validation)
- Syntactic structure: 25% (TAASSC psycholinguistic research)  
- Cohesion/coherence: 20% (Kintsch coherence studies, Coh-Metrix validation)
- Background knowledge: 10% (Medical readability personalization studies)
- Integration synthesis: 5% (Multi-factor analysis research)

**Detailed Final Assessment:**
Provide specific score (1-10), confidence level (High/Medium/Low), and evidence-based justification citing:
- Most problematic sentences and specific difficulty factors
- Vocabulary burden percentage and complexity distribution
- Syntactic complexity patterns and processing demands
- Coherence strengths/weaknesses and flow analysis
- Background knowledge barriers and inference requirements
- Overall difficulty progression and cumulative cognitive load

**Research-Backed Confidence Indicators:**
- High confidence: Consistent patterns across all sentences and factors
- Medium confidence: Clear majority pattern with 1-2 sentence outliers  
- Low confidence: Mixed or contradictory patterns requiring additional analysis

**Final Readability Assessment Scale:**
Based on sentence-level aggregation and research-validated thresholds:

[1-3: Easy/Elementary] - Simple vocabulary, short sentences, clear coherence, minimal background knowledge
[4-6: Medium/Intermediate] - Mixed complexity with manageable difficulty progression
[7-9: Hard/Advanced] - Complex vocabulary, sophisticated syntax, challenging coherence, significant background knowledge required
[10: Very Difficult/Expert] - Sustained complexity across multiple factors creating compound difficulty

transform this result into out of {level}. the final assessment will be a mark (out of {level}). the format of result is as follow: level: (digit)
"""