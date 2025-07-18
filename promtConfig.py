weebit_path = "Dataset/weebit_reextracted.tsv"
onestopenglish_path = "iastate/onestop_english"
UniversalCEFR_path = "UniversalCEFR/readme_en"
model = "deepseek-r1:8b" #(options: deepseek-r1:14b,deepseek-r1:32b,deepseek, qwen, llama)
corpus = 'UniversalCEFR' #(options: weebit, onestopenglish,UniversalCEFR)
level = 5
acc_output_path = "level.csv"
classification_report_path = "classification_report.csv"
data_amount = 40

#hyperparameters
temperature = 0.3
top_p = 0.7

zeroShot_prompt = """
        Analyze the readability of the following text step by step.

        Text: {0}

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
Analyze the readability of the following text using following reasoning.

Text: {0}

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

transform this result into out of {level}. the final assessment will be a mark (out of {1}). the format of result is as follow: level: (digit)
"""

CEFR_prompt = r'''
Analyze the readability of the following text using CEFR(6 levels, A1,A2,B1,B2,C1,C2) Criteria.

Text: {0}

Let's thinks step by step

output the exact CEFR level in the end, Format is as follow:
CEFR Level: X
'''

CEFR_prompt_2 = r'''
Analyze the readability of the following text using the Common European Framework of Reference for Languages (CEFR) criteria. The analysis should determine one of the following six levels: A1, A2, B1, B2, C1, or C2.

Text: {0}

To ensure the most accurate assessment, your step-by-step thinking should evaluate how well a learner at each CEFR level would be able to **understand and do** with the text, considering the following specific linguistic and communicative features:

* **Vocabulary Assessment:**
    * **Lexical Frequency & Range:** Are the words predominantly high-frequency (common everyday words) or do they include lower-frequency, specialized, academic, or idiomatic vocabulary?
    * **Abstractness:** Are concepts concrete and easily visualizable, or do they involve abstract ideas, nuanced meanings, or implied information?
    * **Vocabulary Depth:** Does understanding require distinguishing between subtle shades of meaning or recognizing specific collocations?

* **Grammar & Sentence Structure Assessment:**
    * **Syntactic Complexity:** Are sentences typically short and simple (e.g., SVO structure), or do they frequently employ complex structures such as multiple clauses, passive voice, inversions, or sophisticated conjunctions?
    * **Grammatical Range:** What range of tenses, aspects, moods (e.g., subjunctive), and grammatical structures are used? Is the grammar consistently accurate?

* **Cohesion & Coherence Assessment:**
    * **Logical Organization:** Is the text's structure clear and easy to follow, or does it require the reader to infer relationships between ideas?
    * **Discourse Markers:** Are connections between ideas explicitly signaled with simple linking words, or does it rely on more advanced discourse markers and implicit rhetorical relationships?
    * **Referencing:** Are anaphoric/cataphoric references clear and unambiguous?

* **Pragmatics & Register Assessment:**
    * **Implied Meaning & Nuance:** Is the meaning always explicit, or does the text require the reader to infer implied meanings, understand irony, or recognize cultural references/idioms?
    * **Formality:** Is the language consistently informal, neutral, or does it involve shifts in register and more formal/academic language?
    * **Text Purpose:** What is the communicative purpose of the text (e.g., inform, persuade, entertain, instruct)? How does this impact the linguistic demands?

* **Overall Text Type & Cognitive Demand:**
    * **Genre Familiarity:** Is this a typical text type (e.g., simple notice, personal letter, news report, academic article, literary prose) for the target CEFR level?
    * **Topic Familiarity & Background Knowledge:** Does understanding the text require specific background knowledge or is the topic generally accessible?
    * **Processing Load:** How much cognitive effort is required for a learner at the proposed level to fully comprehend the information, follow arguments, or grasp nuanced points?

Your final output should be the exact CEFR level, like the format as follow.

CEFR Level: X
'''


CEFR_prompt_4 = r"""
You are a Common European Framework of Reference for Languages (CEFR) language assessment expert. I will give you a text, and you will estimate its CEFR level (A1 to C2) using a step-by-step reasoning process based on follow rules.
Rules1:
Identify the reading purpose and text type first, such as Is it reading for information, argument, leisure, instruction, orientation, or correspondence?
Rules2: 
Describe the linguistic features of the text，candidate features include documents length, number of characters per sentence, syllables per sentence and word. Lexical complexity. Average distance between words, parse tree height. etc.
Rules3:
Assess the cognitive demands. Does the reader need to infer meaning? Are there abstract concepts, idioms, or multiple viewpoints?
Rules4:
Compare findings with CEFR reading descriptors, such as, Choose the most appropriate CEFR level (A1 to C2), referencing typical ability ranges (e.g., A2 = short, simple texts; B2 = complex argument with implied meaning).
Give your estimated CEFR level and explain your reasoning clearly.
Text:{0}
After thinking, please give Your final evaluation, the final output should be the exact CEFR level, like the format as follow.
CEFR Level: X
"""

CEFR_prompt_5 = r"""
You are an expert in language proficiency classification based on the Common European Framework of Reference for Languages (CEFR). Your task is to analyze the given text or narrative and using a step-by-step reasoning process to determine the best CEFR level [A1, A2, B1, B2, C1, or C2] based on the CEFR descriptors of reading comprehension of learners below:
A1 - I can understand familiar names, words and very simple sentences, for example on notices and posters or in catalogues. 
A2 - I can read very short, simple texts. I can find specific, predictable information in simple everyday material such as advertisements, prospectuses, menus and timetables and I can understand short simple personal letters.
B1 - I can understand texts that consist mainly of high frequency everyday or job-related language. I can understand the description of events, feelings and wishes in personal letters. 
B2 - I can read articles and reports concerned with contemporary problems in which the writers adopt particular attitudes or viewpoints. I can understand contemporary literary prose.
C1 - I can understand long and complex factual and literary texts, appreciating distinctions of style. I can understand specialised articles and longer technical instructions, even when they do not relate to my field.
C2 - I can read with ease virtually all forms of the written language, including abstract, structurally or linguistically complex texts such as manuals, specialised articles and literary works

The given text is:{0}

Give your estimated CEFR level and explain your reasoning clearly. For example, text=” Overall, this strategy is quite effective at handling non-congestive losses without losing throughput” analyze it like ‘The sentence uses precise vocabulary ("strategy," "effective," "handling") and adverbial modifiers ("quite") to convey exact meaning. The use of "non-congestive" and especially "throughput" demonstrates mastery of highly specific, often domain-specific, terminology that A1,A2,B1,B2 learner can’t understand it. But C2 learners can use specialized lexis appropriately. The sentence efficiently packs a lot of information into a single, well-structured clause with a dependent phrase ("without losing throughput"). There's no simplification or circumlocution. The prepositions ("at handling") and the construction ("without losing") are used flawlessly and idiomatically. The overall tone is formal, academic, or technical, typical of C2 output in discussions of complex topics. So I will give this text a C2 level’

After thinking, please output the result of the given text in the following format:
CEFR Level: <Level>
"""


CEFR_prompt_6 = r"""
Analyze the given text or narrative and using a step-by-step reasoning process to determine the best CEFR level [A1, A2, B1, B2, C1, or C2] based on the CEFR descriptors of reading comprehension of learners below:
A1 - I can understand familiar names, words and very simple sentences, for example on notices and posters or in catalogues. 
A2 - I can read very short, simple texts. I can find specific, predictable information in simple everyday material such as advertisements, prospectuses, menus and timetables and I can understand short simple personal letters.
B1 - I can understand texts that consist mainly of high frequency everyday or job-related language. I can understand the description of events, feelings and wishes in personal letters. 
B2 - I can read articles and reports concerned with contemporary problems in which the writers adopt particular attitudes or viewpoints. I can understand contemporary literary prose.
C1 - I can understand long and complex factual and literary texts, appreciating distinctions of style. I can understand specialised articles and longer technical instructions, even when they do not relate to my field.
C2 - I can read with ease virtually all forms of the written language, including abstract, structurally or linguistically complex texts such as manuals, specialised articles and literary works

The given text is:{0}

Give your estimated CEFR level and explain your reasoning clearly, For example, text=” Overall, this strategy is quite effective at handling non-congestive losses without losing throughput”, your output should be like: ‘The sentence uses precise vocabulary ("strategy," "effective," "handling") and adverbial modifiers ("quite") to convey exact meaning. The use of "non-congestive" and especially "throughput" demonstrates mastery of highly specific, often domain-specific, terminology that A1,A2,B1,B2 learner can’t understand it. But C2 learners can use specialized lexis appropriately. The sentence efficiently packs a lot of information into a single, well-structured clause with a dependent phrase ("without losing throughput"). There's no simplification or circumlocution. The prepositions ("at handling") and the construction ("without losing") are used flawlessly and idiomatically. The overall tone is formal, academic, or technical, typical of C2 output in discussions of complex topics. So I will give this text a C2 level. CEFR Level: C2’

you have 5000 words limitation for your all outputs(include thinking part), especial your thinking part, only 3000 words allowed.

After thinking, please output the result of the given text in the following format:
CEFR Level: <Level>
"""
promptName = CEFR_prompt_6