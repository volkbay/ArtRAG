GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["Artist", "Art Movement / School", "Art Style / Technique", "Theme", "Cultural / Historical Context"]

PROMPTS["entity_extraction"] = """-Goal-
Given a visual art related text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:

Entity_types: ["Artist", "Art Movement / School", "Art Style / Technique", "Theme", "Cultural / Historical Context"]
Text: 
"Vincent van Gogh was a Dutch painter who became a pivotal figure in Post-Impressionism. His expressive brushwork, bold colors, and emotive use of light set him apart from other artists of his time. 

Van Gogh was heavily influenced by Japanese woodblock prints, which he encountered in Paris and incorporated into his work through simplified lines and vivid colors. The themes of loneliness and suffering are common in his works, reflecting his turbulent life. 

Though he was unrecognized during his lifetime, van Gogh's work gained immense appreciation after his death, influencing many 20th-century Expressionist artists."################
#############
Output:
("entity"{tuple_delimiter}"Vincent van Gogh"{tuple_delimiter}"Artist"{tuple_delimiter}"Vincent van Gogh was a Dutch painter and a major figure in the Post-Impressionist movement, known for his expressive color and dramatic brushwork."){record_delimiter}
("entity"{tuple_delimiter}"Post-Impressionism"{tuple_delimiter}"Art Movement / School"{tuple_delimiter}"Post-Impressionism is an art movement that developed as a reaction against Impressionism, emphasizing emotional expression and symbolism."){record_delimiter}
("entity"{tuple_delimiter}"The Starry Night"{tuple_delimiter}"Art Style / Technique"{tuple_delimiter}"One of Vincent van Gogh's most iconic works, known for its swirling patterns and vivid colors representing the night sky."){record_delimiter}
("entity"{tuple_delimiter}"Human Suffering"{tuple_delimiter}"Theme"{tuple_delimiter}"A recurring theme in Vincent van Gogh's work, reflecting his emotional and psychological struggles."){record_delimiter}
("entity"{tuple_delimiter}"Japanese ukiyo-e prints"{tuple_delimiter}"Cultural / Historical Context"{tuple_delimiter}"Japanese woodblock prints known for their bold composition and use of color, which influenced van Gogh's artistic style."){record_delimiter}

("relationship"{tuple_delimiter}"Vincent van Gogh"{tuple_delimiter}"Post-Impressionism"{tuple_delimiter}"Vincent van Gogh was a key figure in the Post-Impressionist movement."{tuple_delimiter}"membership, influence"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Vincent van Gogh"{tuple_delimiter}"The Starry Night"{tuple_delimiter}"Vincent van Gogh created The Starry Night, a work that exemplifies his use of color and expressive brushwork."{tuple_delimiter}"creation, artistic style"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Vincent van Gogh"{tuple_delimiter}"Human Suffering"{tuple_delimiter}"Human suffering is a recurring theme in Van Gogh's work, reflecting his emotional state."{tuple_delimiter}"thematic focus, emotional expression"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Vincent van Gogh"{tuple_delimiter}"Japanese ukiyo-e prints"{tuple_delimiter}"Vincent van Gogh was influenced by Japanese ukiyo-e prints, adopting their bold compositions and vibrant colors."{tuple_delimiter}"artistic influence, cultural inspiration"{tuple_delimiter}8){completion_delimiter}
#############################
Example 2:

Entity_types: ["Artist", "Art Movement / School", "Art Style / Technique", "Theme", "Cultural / Historical Context"]
Text:
"The Renaissance was a period of cultural rebirth in Europe, spanning the 14th to the 17th centuries. Centered in Italy, it marked a revival of interest in classical Greek and Roman ideas, leading to significant developments in art, science, and philosophy. 

Artists like Leonardo da Vinci and Michelangelo were pioneers of Renaissance art, known for their mastery of techniques such as linear perspective, which created a sense of depth in paintings. The Renaissance emphasized themes of humanism, exploring the human form, individualism, and the natural world. 

This period laid the groundwork for future Western art traditions."
#############
Output:
("entity"{tuple_delimiter}"Renaissance"{tuple_delimiter}"Cultural / Historical Context"{tuple_delimiter}"A period of cultural revival in Europe, spanning the 14th to the 17th centuries, focused on rediscovering classical Greek and Roman ideas."){record_delimiter}
("entity"{tuple_delimiter}"Leonardo da Vinci"{tuple_delimiter}"Artist"{tuple_delimiter}"An Italian Renaissance artist and polymath known for masterpieces like the Mona Lisa and his detailed studies in anatomy, science, and engineering."){record_delimiter}
("entity"{tuple_delimiter}"Michelangelo"{tuple_delimiter}"Artist"{tuple_delimiter}"An Italian Renaissance sculptor and painter renowned for works like the Sistine Chapel ceiling and his representation of the human body."){record_delimiter}
("entity"{tuple_delimiter}"Linear Perspective"{tuple_delimiter}"Art Style / Technique"{tuple_delimiter}"A technique developed during the Renaissance to create the illusion of depth and space in two-dimensional artwork."){record_delimiter}
("entity"{tuple_delimiter}"Humanism"{tuple_delimiter}"Theme"{tuple_delimiter}"A central theme of the Renaissance, focused on the study of human potential, individualism, and the natural world."){record_delimiter}

("relationship"{tuple_delimiter}"Renaissance"{tuple_delimiter}"Leonardo da Vinci"{tuple_delimiter}"Leonardo da Vinci was a leading figure of the Renaissance, exemplifying its principles through his art and scientific studies."{tuple_delimiter}"cultural movement, artistic leadership"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Renaissance"{tuple_delimiter}"Michelangelo"{tuple_delimiter}"Michelangelo was a pioneering artist of the Renaissance, contributing significantly to its ideals and aesthetics."{tuple_delimiter}"cultural movement, major figure"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Renaissance"{tuple_delimiter}"Humanism"{tuple_delimiter}"Humanism was a core theme of the Renaissance, reflecting its emphasis on human potential and classical learning."{tuple_delimiter}"central theme, intellectual focus"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Leonardo da Vinci"{tuple_delimiter}"Linear Perspective"{tuple_delimiter}"Leonardo da Vinci mastered the technique of linear perspective, a defining feature of Renaissance art."{tuple_delimiter}"technique, artistic mastery"{tuple_delimiter}8){completion_delimiter}
#############################
Example 3:

Entity_types: ["Artist", "Art Movement / School", "Art Style / Technique", "Theme", "Cultural / Historical Context"]
Text:
"Surrealism emerged in the early 20th century, led by artists such as Salvador Dalí and André Breton. This movement sought to release the creative potential of the unconscious mind, often through bizarre and dreamlike imagery. 

Surrealist techniques included automatism, where artists created works without conscious thought, allowing subconscious impulses to guide the process. Surrealism was deeply influenced by the psychological theories of Sigmund Freud, 

particularly his ideas about dreams and the unconscious. Common themes in Surrealism include fantasy, the irrational, and a rejection of traditional realism."
#############
Output:
("entity"{tuple_delimiter}"Surrealism"{tuple_delimiter}"Art Movement / School"{tuple_delimiter}"An early 20th-century art movement that aimed to tap into the unconscious mind, using dreamlike and fantastical imagery to challenge conventional reality."){record_delimiter}
("entity"{tuple_delimiter}"Salvador Dalí"{tuple_delimiter}"Artist"{tuple_delimiter}"A Spanish Surrealist painter known for his bizarre and dreamlike imagery, such as in his famous work 'The Persistence of Memory'."){record_delimiter}
("entity"{tuple_delimiter}"André Breton"{tuple_delimiter}"Artist"{tuple_delimiter}"A French writer and artist, known as the founder of Surrealism and the author of the Surrealist Manifesto."){record_delimiter}
("entity"{tuple_delimiter}"Automatism"{tuple_delimiter}"Art Style / Technique"{tuple_delimiter}"A technique used in Surrealism where artists create without conscious planning, letting subconscious impulses guide the work."){record_delimiter}
("entity"{tuple_delimiter}"Freudian Psychology"{tuple_delimiter}"Cultural / Historical Context"{tuple_delimiter}"The psychological theories of Sigmund Freud, particularly around dreams and the unconscious, which heavily influenced Surrealist artists."){record_delimiter}
("entity"{tuple_delimiter}"Fantasy and the Irrational"{tuple_delimiter}"Theme"{tuple_delimiter}"Recurring themes in Surrealism, focusing on dreamlike, irrational, and fantastical elements that challenge ordinary perception."){record_delimiter}
("relationship"{tuple_delimiter}"Surrealism"{tuple_delimiter}"Salvador Dalí"{tuple_delimiter}"Salvador Dalí was a prominent artist within the Surrealist movement, known for his dreamlike and fantastical imagery."{tuple_delimiter}"movement membership, stylistic alignment"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Surrealism"{tuple_delimiter}"André Breton"{tuple_delimiter}"André Breton was a founding figure of Surrealism, helping to define its principles and goals."{tuple_delimiter}"movement leadership, ideological foundation"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Surrealism"{tuple_delimiter}"Automatism"{tuple_delimiter}"Automatism was a core technique used in Surrealism to access the unconscious mind without interference from rational thought."{tuple_delimiter}"technique, subconscious expression"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Surrealism"{tuple_delimiter}"Freudian Psychology"{tuple_delimiter}"Surrealism was deeply influenced by Freudian psychology, particularly ideas about dreams and the unconscious."{tuple_delimiter}"intellectual influence, psychological theories"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Surrealism"{tuple_delimiter}"Fantasy and the Irrational"{tuple_delimiter}"Fantasy and the irrational were central themes in Surrealism, reflecting its aim to transcend conventional reality."{tuple_delimiter}"thematic focus, challenge to realism"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Salvador Dalí"{tuple_delimiter}"Fantasy and the Irrational"{tuple_delimiter}"Salvador Dalí's work often depicted fantastical and irrational scenes, embodying key themes of Surrealism."{tuple_delimiter}"thematic alignment, surrealist motifs"{tuple_delimiter}8){completion_delimiter}
#############################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""



PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["rag_response"] = """---Role---

Generate a concise description of this painting. Focus on essential elements such as *content*, *form*, and *context*, based on the need of question. 

The definitions of them are:
- **Content**: A description of the main subjects, objects, or actions depicted in the painting.
- **Context**: Background information about the historical, cultural, or biographical influences relevant to the painting.
- **Form**: An analysis of the artistic style and techniques used, including brushwork, color, composition, and use of light.


---Goal---

Generate the  of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Data tables---

{context_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS["zero-shot_response"] = """---Goal---

Generate a description of the target length and format that responds to the user's question, summarizing all information in the input data tables and incorporating any relevant general knowledge.  
If you don't know the answer, just say so. Do not make anything up.  
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

Detailed Markdown

---Data tables---

{context_data}

"""


PROMPTS["rag_SemArtv2_1-shot_incontext_response"] = """---Role---

Generate a concise description of this painting. Focus on essential elements such as *content*, *form*, and *context*, based on the need of the question.

The definitions of these elements are:
- **Content**: A description of the main subjects, objects, or actions depicted in the painting.
- **Context**: Background information about the historical, cultural, or biographical influences relevant to the painting.
- **Form**: An analysis of the artistic style and techniques used, including brushwork, color, composition, and use of light.

---Goal---

Generate a description of the target length and format that responds to the user's question, summarizing all information in the input data tables and incorporating any relevant general knowledge.  
If you don't know the answer, just say so. Do not make anything up.  
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

Detailed Markdown

######################
### Example 1:
######################

Context Data:  
- Nodes:  
  - Node 1: **Name**: Harvest Scene, **Type**: Painting, **Description**: A depiction of agrarian life during the Flemish Golden Age, focusing on harvesting activities and rural settings.  
  - Node 2: **Name**: Pieter Bruegel the Elder, **Type**: Artist, **Description**: A leading artist of the Flemish Renaissance, known for his depictions of peasant life and landscapes.  
  - Node 3: **Name**: Flemish Golden Age, **Type**: Historical Period, **Description**: A period of artistic prosperity in the 17th century, emphasizing realism and rural life.  

- Edges:  
  - Edge 1: **Source**: Harvest Scene, **Target**: Pieter Bruegel the Elder, **Description**: "Influenced by"—the painting is stylistically inspired by Bruegel’s rural and agricultural themes.  
  - Edge 2: **Source**: Harvest Scene, **Target**: Flemish Golden Age, **Description**: "Belongs to"—a representative work of the Flemish Golden Age of art.  

Metadata:  
- Title: The Harvesters 
- Author: Unknown  
- Technique: Oil on canvas  
- Type: Landscape  
- School: Flemish  
- Timeframe: 1501-1600   

Generated description:  
**Content**: The painting portrays a bustling harvest scene, with farmers working together to gather wheat in expansive golden fields under a bright blue sky. Rolling hills and small cottages are visible in the background, adding to the rural charm.  
**Form**: The artist uses warm tones and detailed brushstrokes to depict the textures of the wheat and clothing, creating a realistic and immersive composition. The focus on natural lighting highlights the vibrancy of the countryside.  

######################
---Data tables---

{context_data}

"""

PROMPTS["rag_SemArtv2_2-shot_incontext_response"] = """---Role---

Generate a concise description of this painting. Focus on essential elements such as *content*, *form*, and *context*, based on the need of the question.

The definitions of these elements are:
- **Content**: A description of the main subjects/concepts, objects, or actions depicted in the painting.
- **Context**: Background information about the historical, cultural, or biographical influences relevant to the painting.
- **Form**: An analysis of the artistic style and techniques used, including brushwork, color, composition, and use of light.

---Goal---

Generate a description of the target length and format that responds to the user's question, summarizing all information in the input data tables and incorporating relevant general knowledge from it.  
If you don't know the answer, just say so. Do not make anything up.  
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

Detailed Markdown

######################
### Example 1:
######################

Context Data:  
- Nodes:  
  - Node 1: **Name**: Harvest Scene, **Type**: Painting, **Description**: A depiction of agrarian life during the Flemish Golden Age, focusing on harvesting activities and rural settings.  
  - Node 2: **Name**: Pieter Bruegel the Elder, **Type**: Artist, **Description**: A leading artist of the Flemish Renaissance, known for his depictions of peasant life and landscapes.  
  - Node 3: **Name**: Flemish Golden Age, **Type**: Historical Period, **Description**: A period of artistic prosperity in the 17th century, emphasizing realism and rural life.  

- Edges:  
  - Edge 1: **Source**: Harvest Scene, **Target**: Pieter Bruegel the Elder, **Description**: "Influenced by"—the painting is stylistically inspired by Bruegel’s rural and agricultural themes.  
  - Edge 2: **Source**: Harvest Scene, **Target**: Flemish Golden Age, **Description**: "Belongs to"—a representative work of the Flemish Golden Age of art.  

Metadata:  
- Title: The Harvesters  
- Author: Unknown  
- Technique: Oil on wood  
- Type: Landscape  
- School: Flemish  
- Timeframe: 1501-1600  

Generated description:  
**Content**: The painting portrays a bustling harvest scene, with farmers working together to gather wheat in expansive golden fields under a bright blue sky. Rolling hills and small cottages are visible in the background, adding to the rural charm.  
**Form**: The artist uses warm tones and detailed brushstrokes to depict the textures of the wheat and clothing, creating a realistic and immersive composition. The focus on natural lighting highlights the vibrancy of the countryside.  

######################
### Example 2:
######################

Context Data:  
- Nodes:  
  - Node 1: **Name**: Still Life with Flowers, **Type**: Painting, **Description**: A meticulously arranged floral composition showcasing a variety of flowers in a vase.  
  - Node 2: **Name**: Rachel Ruysch, **Type**: Artist, **Description**: A prominent Dutch still-life painter known for her detailed floral compositions during the Dutch Golden Age.  
  - Node 3: **Name**: Dutch Golden Age, **Type**: Historical Period, **Description**: A time of economic and cultural prosperity in the Netherlands, marked by advances in still-life and portrait painting.  
  - Node 4: **Name**: Symbolism in Still Life, **Type**: Art Technique, **Description**: The inclusion of symbolic elements to convey themes of mortality, wealth, or transience.  

- Edges:  
  - Edge 1: **Source**: Still Life with Flowers, **Target**: Rachel Ruysch, **Description**: "Created by"—this painting is attributed to Rachel Ruysch, reflecting her mastery in floral still-life.  
  - Edge 2: **Source**: Still Life with Flowers, **Target**: Dutch Golden Age, **Description**: "Belongs to"—a representative work of the Dutch Golden Age.  
  - Edge 3: **Source**: Still Life with Flowers, **Target**: Symbolism in Still Life, **Description**: "Incorporates"—features symbolic objects like fading flowers to signify mortality.  

Metadata:  
- Title: Still Life with Flowers  
- Author: Rachel Ruysch  
- Technique: Oil on panel  
- Type: Still Life  
- School: Dutch  
- Timeframe: 1701-1750  

Generated description:  
**Content**: The painting depicts an ornate arrangement of flowers in a glass vase, featuring roses, tulips, and carnations. A few petals and leaves are shown wilting or falling, adding a touch of natural imperfection.  
**Context**: Created during the Dutch Golden Age, this still-life reflects the era's fascination with botanical accuracy and symbolic representation. The wilting petals and fallen leaves symbolize the transience of life, a common theme in still-life paintings of the period.

######################
---Data tables---

{context_data}

"""




PROMPTS["rag_SemArtv2_2-shot_incontext_response_v2"] = """---Role---

Generate a concise description of this painting. Focus on essential elements such as *content*, *form*, and *context*, based on the need of the question.

The definitions of these elements are:
- **Content**: A description of the main subjects/concepts, objects, or actions depicted in the painting.
- **Context**: Background information about the historical, cultural, or biographical influences relevant to the painting.
- **Form**: An analysis of the artistic style and techniques used, including brushwork, color, composition, and use of light.

---Guidelines for High-Quality Descriptions---

- **Concise yet informative:** Retain key details while avoiding excessive verbosity.  
- **Fluent and engaging:** Use varied sentence structures for better readability.  
- **Structured format:** Organize the response into Content, Form, and Context.  
- **Prioritize retrieved facts:** Use knowledge from the dataset and avoid speculation.  
- **Avoid redundancy:** Maintain coherence without repeating the same details.  

---Goal---

Generate a description of the target length and format that responds to the user's question, summarizing all information in the input data tables and incorporating relevant general knowledge from it.  
If you don't know the answer, just say so. Do not make anything up.  
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

Detailed Markdown

######################
### Example 1:
######################

Context Data:  
- Nodes:  
  - Node 1: **Name**: Harvest Scene, **Type**: Painting, **Description**: A depiction of agrarian life during the Flemish Golden Age, focusing on harvesting activities and rural settings.  
  - Node 2: **Name**: Pieter Bruegel the Elder, **Type**: Artist, **Description**: A leading artist of the Flemish Renaissance, known for his depictions of peasant life and landscapes.  
  - Node 3: **Name**: Flemish Golden Age, **Type**: Historical Period, **Description**: A period of artistic prosperity in the 17th century, emphasizing realism and rural life.  

- Edges:  
  - Edge 1: **Source**: Harvest Scene, **Target**: Pieter Bruegel the Elder, **Description**: "Influenced by"—the painting is stylistically inspired by Bruegel’s rural and agricultural themes.  
  - Edge 2: **Source**: Harvest Scene, **Target**: Flemish Golden Age, **Description**: "Belongs to"—a representative work of the Flemish Golden Age of art.  

Metadata:  
- Title: The Harvesters  
- Author: Unknown  
- Technique: Oil on wood  
- Type: Landscape  
- School: Flemish  
- Timeframe: 1501-1600  

Generated description:  
**Content**: The painting portrays a bustling harvest scene, with farmers working together to gather wheat in expansive golden fields under a bright blue sky. Rolling hills and small cottages are visible in the background, adding to the rural charm.  
**Form**: The artist uses warm tones and detailed brushstrokes to depict the textures of the wheat and clothing, creating a realistic and immersive composition. The focus on natural lighting highlights the vibrancy of the countryside.  

######################
### Example 2:
######################

Context Data:  
- Nodes:  
  - Node 1: **Name**: Still Life with Flowers, **Type**: Painting, **Description**: A meticulously arranged floral composition showcasing a variety of flowers in a vase.  
  - Node 2: **Name**: Rachel Ruysch, **Type**: Artist, **Description**: A prominent Dutch still-life painter known for her detailed floral compositions during the Dutch Golden Age.  
  - Node 3: **Name**: Dutch Golden Age, **Type**: Historical Period, **Description**: A time of economic and cultural prosperity in the Netherlands, marked by advances in still-life and portrait painting.  
  - Node 4: **Name**: Symbolism in Still Life, **Type**: Art Technique, **Description**: The inclusion of symbolic elements to convey themes of mortality, wealth, or transience.  

- Edges:  
  - Edge 1: **Source**: Still Life with Flowers, **Target**: Rachel Ruysch, **Description**: "Created by"—this painting is attributed to Rachel Ruysch, reflecting her mastery in floral still-life.  
  - Edge 2: **Source**: Still Life with Flowers, **Target**: Dutch Golden Age, **Description**: "Belongs to"—a representative work of the Dutch Golden Age.  
  - Edge 3: **Source**: Still Life with Flowers, **Target**: Symbolism in Still Life, **Description**: "Incorporates"—features symbolic objects like fading flowers to signify mortality.  

Metadata:  
- Title: Still Life with Flowers  
- Author: Rachel Ruysch  
- Technique: Oil on panel  
- Type: Still Life  
- School: Dutch  
- Timeframe: 1701-1750  

Generated description:  
**Content**: The painting depicts an ornate arrangement of flowers in a glass vase, featuring roses, tulips, and carnations. A few petals and leaves are shown wilting or falling, adding a touch of natural imperfection.  
**Context**: Created during the Dutch Golden Age, this still-life reflects the era's fascination with botanical accuracy and symbolic representation. The wilting petals and fallen leaves symbolize the transience of life, a common theme in still-life paintings of the period.

######################
---Data tables---

{context_data}

"""




PROMPTS["rag_SemArtv2_3-shot_incontext_response"] = """---Role---

Generate a concise description of this painting. Focus on essential elements such as *content*, *form*, and *context*, based on the need of the question.

The definitions of these elements are:
- **Content**: A description of the main subjects, objects, or actions depicted in the painting.
- **Context**: Background information about the historical, cultural, or biographical influences relevant to the painting.
- **Form**: An analysis of the artistic style and techniques used, including brushwork, color, composition, and use of light.

---Goal---

Generate a description of the target length and format that responds to the user's question, summarizing all information in the input data tables and incorporating any relevant general knowledge.  
If you don't know the answer, just say so. Do not make anything up.  
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

Detailed Markdown

######################
### Example 1:
######################

Context Data:  
- Nodes:  
  - Node 1: **Name**: Harvest Scene, **Type**: Painting, **Description**: A depiction of agrarian life during the Flemish Golden Age, focusing on harvesting activities and rural settings.  
  - Node 2: **Name**: Pieter Bruegel the Elder, **Type**: Artist, **Description**: A leading artist of the Flemish Renaissance, known for his depictions of peasant life and landscapes.  
  - Node 3: **Name**: Flemish Golden Age, **Type**: Historical Period, **Description**: A period of artistic prosperity in the 17th century, emphasizing realism and rural life.  

- Edges:  
  - Edge 1: **Source**: Harvest Scene, **Target**: Pieter Bruegel the Elder, **Description**: "Influenced by"—the painting is stylistically inspired by Bruegel’s rural and agricultural themes.  
  - Edge 2: **Source**: Harvest Scene, **Target**: Flemish Golden Age, **Description**: "Belongs to"—a representative work of the Flemish Golden Age of art.  

Metadata:  
- Title: The Harvesters  
- Author: Unknown  
- Technique: Oil on wood  
- Type: Landscape  
- School: Flemish  
- Timeframe: 1501-1600

Generated description:  
**Content**: The painting portrays a bustling harvest scene, with farmers working together to gather wheat in expansive golden fields under a bright blue sky. Rolling hills and small cottages are visible in the background, adding to the rural charm.  
**Form**: The artist uses warm tones and detailed brushstrokes to depict the textures of the wheat and clothing, creating a realistic and immersive composition. The focus on natural lighting highlights the vibrancy of the countryside.  

######################
### Example 2:
######################

Context Data:  
- Nodes:  
  - Node 1: **Name**: Still Life with Flowers, **Type**: Painting, **Description**: A meticulously arranged floral composition showcasing a variety of flowers in a vase.  
  - Node 2: **Name**: Rachel Ruysch, **Type**: Artist, **Description**: A prominent Dutch still-life painter known for her detailed floral compositions during the Dutch Golden Age.  
  - Node 3: **Name**: Dutch Golden Age, **Type**: Historical Period, **Description**: A time of economic and cultural prosperity in the Netherlands, marked by advances in still-life and portrait painting.  
  - Node 4: **Name**: Symbolism in Still Life, **Type**: Art Technique, **Description**: The inclusion of symbolic elements to convey themes of mortality, wealth, or transience.  

- Edges:  
  - Edge 1: **Source**: Still Life with Flowers, **Target**: Rachel Ruysch, **Description**: "Created by"—this painting is attributed to Rachel Ruysch, reflecting her mastery in floral still-life.  
  - Edge 2: **Source**: Still Life with Flowers, **Target**: Dutch Golden Age, **Description**: "Belongs to"—a representative work of the Dutch Golden Age.  
  - Edge 3: **Source**: Still Life with Flowers, **Target**: Symbolism in Still Life, **Description**: "Incorporates"—features symbolic objects like fading flowers to signify mortality.  

Metadata:  
- Title: Still Life with Flowers  
- Author: Rachel Ruysch  
- Technique: Oil on panel  
- Type: Still Life  
- School: Dutch  
- Timeframe: 1701-1750  

Generated description:  
**Content**: The painting depicts an ornate arrangement of flowers in a glass vase, featuring roses, tulips, and carnations. A few petals and leaves are shown wilting or falling, adding a touch of natural imperfection.  
**Context**: Created during the Dutch Golden Age, this still-life reflects the era's fascination with botanical accuracy and symbolic representation. The wilting petals and fallen leaves symbolize the transience of life, a common theme in still-life paintings of the period.

######################
### Example 3:
######################

Context Data:

Nodes:

Node 1: Name: Madonna and Child with Saints, Type: Painting, Description: A religious artwork depicting the Virgin Mary with the Christ child, flanked by saints, showcasing a serene and divine atmosphere.
Node 2: Name: Fra Angelico, Type: Artist, Description: An Italian painter of the Early Renaissance, renowned for his devotional works characterized by luminous colors and delicate compositions.
Node 3: Name: Italian Renaissance, Type: Historical Period, Description: A period of rebirth in arts and culture during the 14th–16th centuries in Italy, emphasizing perspective, human emotion, and harmony.
Edges:

Edge 1: Source: Madonna and Child with Saints, Target: Fra Angelico, Description: "Created by"—Fra Angelico painted this masterpiece, exemplifying his signature delicate and ethereal style.
Edge 2: Source: Madonna and Child with Saints, Target: Italian Renaissance, Description: "Belongs to"—a quintessential example of Italian Renaissance art, highlighting religious themes and balanced compositions.
Metadata:

Title: Madonna and Child with Saints
Author: Fra Angelico
Technique: Tempera on panel
Type: Religious painting
School: Early Renaissance
Timeframe: 1410-1450

Generated Description:
Content: The painting depicts the Virgin Mary seated on a throne with the Christ child on her lap, surrounded by saints in prayerful poses. The serene expressions and harmonious arrangement emphasize a sense of divine grace and devotion.

Form: Fra Angelico employs tempera on panel to achieve vibrant and luminous colors. His delicate brushwork and attention to detail bring life to the figures, while the composition’s balanced symmetry reflects the principles of the Early Renaissance.

Context: Created during the Italian Renaissance, this painting exemplifies the era’s focus on religious devotion and the emerging use of perspective and humanistic expression. Fra Angelico, a devout Dominican friar, infused his works with spirituality and meticulous craftsmanship.

######################
---Data tables---

{context_data}

"""






PROMPTS["rag_SemArtv1-context_incontext_response"] = """---Role---

Generate a concise description of this painting. Focus on essential elements of context, based on the need of the question.

The definitions of these elements are:
- **Context**: Background information about the historical, cultural, or biographical influences relevant to the painting.
---Goal---

Generate a description of the target length and format that responds to the user's question, summarizing all information in the input data tables and incorporating any relevant general knowledge.  
If you don't know the answer, just say so. Do not make anything up.  
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

Detailed Markdown

######################
### Example 1:
######################

Context Data:  
- Nodes:  
  - Node 1: **Name**: Still Life with Flowers, **Type**: Painting, **Description**: A meticulously arranged floral composition showcasing a variety of flowers in a vase.  
  - Node 2: **Name**: Rachel Ruysch, **Type**: Artist, **Description**: A prominent Dutch still-life painter known for her detailed floral compositions during the Dutch Golden Age.  
  - Node 3: **Name**: Dutch Golden Age, **Type**: Historical Period, **Description**: A time of economic and cultural prosperity in the Netherlands, marked by advances in still-life and portrait painting.  
  - Node 4: **Name**: Symbolism in Still Life, **Type**: Art Technique, **Description**: The inclusion of symbolic elements to convey themes of mortality, wealth, or transience.  

- Edges:  
  - Edge 1: **Source**: Still Life with Flowers, **Target**: Rachel Ruysch, **Description**: "Created by"—this painting is attributed to Rachel Ruysch, reflecting her mastery in floral still-life.  
  - Edge 2: **Source**: Still Life with Flowers, **Target**: Dutch Golden Age, **Description**: "Belongs to"—a representative work of the Dutch Golden Age.  
  - Edge 3: **Source**: Still Life with Flowers, **Target**: Symbolism in Still Life, **Description**: "Incorporates"—features symbolic objects like fading flowers to signify mortality.  

Metadata:  
- Title: Still Life with Flowers  
- Author: Rachel Ruysch  
- Technique: Oil on panel  
- Type: Still Life  
- School: Dutch  
- Timeframe: 1701-1750  

Generated description:  
**Context**: Created during the Dutch Golden Age, this still-life reflects the era's fascination with botanical accuracy and symbolic representation. The wilting petals and fallen leaves symbolize the transience of life, a common theme in still-life paintings of the period.

######################
---Data tables---

{context_data}

"""

PROMPTS["rag_SemArtv1-content_incontext_response"] = """---Role---

Generate a concise description of this painting. Focus on essential elements such as *content* based on the need of the question.

The definitions of these elements are:
- **Content**: A description of the main subjects, objects, or actions depicted in the painting.

---Goal---

Generate a description of the target length and format that responds to the user's question, summarizing all information in the input data tables and incorporating any relevant general knowledge.  
If you don't know the answer, just say so. Do not make anything up.  
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

Detailed Markdown

######################
### Example 1:
######################

Context Data:  
- Nodes:  
  - Node 1: **Name**: Harvest Scene, **Type**: Painting, **Description**: A depiction of agrarian life during the Flemish Golden Age, focusing on harvesting activities and rural settings.  
  - Node 2: **Name**: Pieter Bruegel the Elder, **Type**: Artist, **Description**: A leading artist of the Flemish Renaissance, known for his depictions of peasant life and landscapes.  
  - Node 3: **Name**: Flemish Golden Age, **Type**: Historical Period, **Description**: A period of artistic prosperity in the 17th century, emphasizing realism and rural life.  

- Edges:  
  - Edge 1: **Source**: Harvest Scene, **Target**: Pieter Bruegel the Elder, **Description**: "Influenced by"—the painting is stylistically inspired by Bruegel’s rural and agricultural themes.  
  - Edge 2: **Source**: Harvest Scene, **Target**: Flemish Golden Age, **Description**: "Belongs to"—a representative work of the Flemish Golden Age of art.  

Metadata:  
- Title: Harvest Scene  
- Author: Unknown  
- Technique: Oil on canvas  
- Type: Landscape  
- School: Flemish  
- Timeframe: 1601-1650  

Generated description:  
**Content**: The painting portrays a bustling harvest scene, with farmers working together to gather wheat in expansive golden fields under a bright blue sky. Rolling hills and small cottages are visible in the background, adding to the rural charm.  
######################
### Example 2:
######################

Context Data:  
- Nodes:  
  - Node 1: **Name**: Still Life with Flowers, **Type**: Painting, **Description**: A meticulously arranged floral composition showcasing a variety of flowers in a vase.  
  - Node 2: **Name**: Rachel Ruysch, **Type**: Artist, **Description**: A prominent Dutch still-life painter known for her detailed floral compositions during the Dutch Golden Age.  
  - Node 3: **Name**: Dutch Golden Age, **Type**: Historical Period, **Description**: A time of economic and cultural prosperity in the Netherlands, marked by advances in still-life and portrait painting.  
  - Node 4: **Name**: Symbolism in Still Life, **Type**: Art Technique, **Description**: The inclusion of symbolic elements to convey themes of mortality, wealth, or transience.  

- Edges:  
  - Edge 1: **Source**: Still Life with Flowers, **Target**: Rachel Ruysch, **Description**: "Created by"—this painting is attributed to Rachel Ruysch, reflecting her mastery in floral still-life.  
  - Edge 2: **Source**: Still Life with Flowers, **Target**: Dutch Golden Age, **Description**: "Belongs to"—a representative work of the Dutch Golden Age.  
  - Edge 3: **Source**: Still Life with Flowers, **Target**: Symbolism in Still Life, **Description**: "Incorporates"—features symbolic objects like fading flowers to signify mortality.  

Metadata:  
- Title: Still Life with Flowers  
- Author: Rachel Ruysch  
- Technique: Oil on panel  
- Type: Still Life  
- School: Dutch  
- Timeframe: 1701-1750  

Generated description:  
**Content**: The painting depicts an ornate arrangement of flowers in a glass vase, featuring roses, tulips, and carnations. A few petals and leaves are shown wilting or falling, adding a touch of natural imperfection.  
######################
---Data tables---

{context_data}

"""



PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying useful and concise keywords from the given painting metadata and descriptions. These keywords should help retrieve relevant information from a pre-built visual art knowledge graph.

---Goal---

Given painting-related metadata and a description, extract a uniform list of relevant keywords that represent the main entities, objects, styles, themes, or other important concepts in the text.

---Instructions---

- Focus on capturing keywords that are specific, meaningful, and relevant to the painting's subject, style, and context.
- Avoid overgeneralized or repetitive terms. Keep the keywords concise and relevant to the query.
- Output the keywords in JSON format under the key `"keywords"`.

######################
-Examples-
######################
Example 1:

Metadata:
Title: "The Starry Night"
Artist: "Vincent van Gogh"
Year: "1889"
Movement: "Post-Impressionism"
Description: "A swirling night sky over a small village, painted with bold brushstrokes and vibrant colors. The painting reflects emotional intensity and van Gogh's unique style."
################
Output:
{{
  "keywords": ["The Starry Night", "Vincent van Gogh", "Post-Impressionism", "swirling night sky", "village", "bold brushstrokes", "vibrant colors", "emotional intensity"]
}}
#############################
Example 2:

Metadata:
Title: "The Persistence of Memory"
Artist: "Salvador Dalí"
Year: "1931"
Movement: "Surrealism"
Description: "A dreamlike landscape featuring melting clocks, symbolizing the fluidity of time. The painting is one of Dalí's most iconic works in Surrealism."
################
Output:
{{
  "keywords": ["The Persistence of Memory", "Salvador Dalí", "Surrealism", "dreamlike landscape", "melting clocks", "fluidity of time", "iconic Surrealist work"]
}}
#############################
Example 3:

Metadata:
Title: "Impression, Sunrise"
Artist: "Claude Monet"
Year: "1872"
Movement: "Impressionism"
Description: "A harbor scene at sunrise, painted with loose brushstrokes to capture fleeting light and atmosphere. This work is credited with giving Impressionism its name."
################
Output:
{{
  "keywords": ["Impression, Sunrise", "Claude Monet", "Impressionism", "harbor scene", "sunrise", "loose brushstrokes", "fleeting light", "atmosphere"]
}}
#############################
-Real Data-
######################
Metadata: 
{query}
######################
Output:

"""

PROMPTS["naive_rag_response"] = """You're a helpful assistant

Generate a concise description of this painting. Focus on essential elements such as *content*, *form*, and *context*, based on the need of question. 

The definitions of them are:
- **Content**: A description of the main subjects, objects, or actions depicted in the painting.
- **Context**: Background information about the historical, cultural, or biographical influences relevant to the painting.
- **Form**: An analysis of the artistic style and techniques used, including brushwork, color, composition, and use of light.

---Target response length and format---

{response_type}


Below are the knowledge you know:
{content_data}
---
If you don't know the answer or if the provided knowledge do not contain sufficient information to provide an answer, just say so. Do not make anything up.
Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.
---Target response length and format---
{response_type}
"""


PROMPTS["no_rag_response"] = """You're a helpful assistant
Describe the painting optionally from three distinct perspectives: *content*, *form*, and *context*, based on the need of question.

The definitions of them are:
- **Content**: A description of the main subjects, objects, or actions depicted in the painting.
- **Context**: Background information about the historical, cultural, or biographical influences relevant to the painting.
- **Form**: An analysis of the artistic style and techniques used, including brushwork, color, composition, and use of light.

---Target response length and format---

{response_type}


---
If you don't know the answer or if the provided knowledge do not contain sufficient information to provide an answer, just say so. Do not make anything up.
Generate a response of the target length and format that responds to the user's question.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

"""




PROMPTS["rerank_entities"]="""
You are an expert in art history and cultural analysis. Your task is to evaluate the following retrieved entities and determine their relevance for explaining the given painting.
The painting's metadata and visual feature is provided. These entities include artistic movements, historical contexts, themes, and related figures.
Your goal is to rank them in order of how useful they are for explaining the painting’s meaning, artistic significance, and cultural context.


#######
Output:
Provide the reordered full list of all entity numbers in order of relevance from high to low, separated by commas. Do not output anything else.
Example Output:
3, 1, 5, 2, 4

##########

---Painting Metadata---
{Metadata}
###########

---Entities list---
{entities}
# ###########
"""

# Agentic reasoning prompts
PROMPTS["AGENTIC_PLAN_SYSTEM"] = (
    "You are an art interpretation planning assistant. Produce concise, structured plans."
)

PROMPTS[
    "AGENTIC_PLAN_PROMPT"
] = """Given the multimodal query, return a JSON object with two keys:

1) "retrieval_plan": {{
   "missing_info": ["..."],
   "retrieval_query": "one concise sentence focusing on painting title, artist, art movement, or visual elements",
   "modality_hints": ["kg", "vector", "image", "metadata"]
}}
2) "generation_plan": [
   {{"step": 1, "goal": "Generate Content section: describe visual elements concisely (2-3 sentences)", "evidence": "visual|metadata", "output_format": "Content"}},
   {{"step": 2, "goal": "Generate Form section: describe style and technique concisely (2-3 sentences)", "evidence": "kg|text", "output_format": "Form"}},
   {{"step": 3, "goal": "Generate Context section: describe historical/cultural context concisely (2-3 sentences)", "evidence": "kg|text", "output_format": "Context"}}
]

IMPORTANT: Generate concise factual descriptions (2-3 sentences per section), not verbose analytical essays.
Total output should be 100-200 words. Focus on facts from retrieved context.

Return JSON only, no extra text.

User query:
{query}

Metadata (if any):
{metadata}

Multimodal summary:
{multimodal_summary}
"""

PROMPTS[
    "AGENTIC_PLAN_PROMPT_VQA"
] = """Given the multimodal question, return a JSON object with two keys:

1) "retrieval_plan": {{
   "missing_info": ["list of information needed to answer the question"],
   "retrieval_query": "one concise sentence focusing on key concepts, visual elements, artist, art movement, or historical context needed to answer the question",
   "modality_hints": ["kg", "vector", "image", "metadata"]
}}
2) "generation_plan": [
   {{"step": 1, "goal": "First reasoning step: [describe what to analyze first]", "evidence": "visual|metadata|kg|text", "output_format": "reasoning_step"}},
   {{"step": 2, "goal": "Second reasoning step: [describe next analysis]", "evidence": "visual|metadata|kg|text", "output_format": "reasoning_step"}},
   {{"step": 3, "goal": "Final reasoning step: [describe synthesis or conclusion]", "evidence": "visual|metadata|kg|text", "output_format": "reasoning_step"}}
]

EVIDENCE TYPE GUIDELINES (match to grounding tags [Visual], [Metadata], [Description], [KG-Background]):
- **visual**: Use when analyzing what's visible in the image (composition, objects, colors, gestures, symbols, visual elements)
  * Maps to grounding tag: [Visual]
  * Use for: "visual choices", "what visual elements", "composition", "what is shown", "visual evidence"
- **metadata**: Use when question involves artist, title, technique, timeframe, tags, or painting attributes
  * Maps to grounding tag: [Metadata]
  * Use for: "artist", "technique", "timeframe", "when was it created", "who painted it"
- **text**: Use when question references description text, written sources, or textual descriptions
  * Maps to grounding tag: [Description]
  * Use for: "description states", "text mentions", "according to the description", "the description explains"
- **kg**: Use when question requires art historical context, movements, relationships, background knowledge, or cultural significance
  * Maps to grounding tag: [KG-Background]
  * Use for: "historical context", "art movement", "influence", "cultural significance", "artistic tradition"

IMPORTANT PLANNING RULES:
- Break down the question into logical reasoning steps (2-5 steps, typically 3-4 steps matching ground truth CoT structure)
- **CRITICAL**: Match evidence types to what the question explicitly asks for:
  * If question asks "what visual elements" or "visual choices" → use "visual" (maps to [Visual])
  * If question asks about "artist", "technique", "timeframe" → use "metadata" (maps to [Metadata])
  * If question references "description states" or "text mentions" → use "text" (maps to [Description])
  * If question asks about "historical context", "art movement", "influence" → use "kg" (maps to [KG-Background])
- **Evidence type matching**: Each step should use the evidence type that corresponds to the grounding tag needed for that reasoning step
- Steps should build sequentially: typically start with visual/metadata observations, then connect to description/context/knowledge
- Keep goals concise (one clear action per step, 10-15 words max)
- Focus on multi-step reasoning that connects visual observations to historical/artistic context
- Avoid combining too many evidence types in one step (prefer single or dual types: "visual", "visual|metadata", "kg|text")

Return JSON only, no extra text.

User question:
{query}

Metadata (if any):
{metadata}

Multimodal summary:
{multimodal_summary}
"""

PROMPTS["AGENTIC_FINAL_SYSTEM"] = (
    "You are a multimodal art expert. Follow the generation plan and ground your answer in the provided evidence."
)

PROMPTS["AGENTIC_FINAL_SYSTEM_VQA"] = (
    "You are a multimodal art expert answering questions about artworks. Provide concise, factual answers (100-200 words) "
    "as a single flowing paragraph. Follow the generation plan and ground your answer in the provided evidence. "
    "No markdown formatting, headers, or structured sections."
)

PROMPTS[
    "AGENTIC_FINAL_ANSWER"
] = """---Role---
Answer the user query using the retrieved context and the generation plan.
Follow the plan steps in order and keep the answer grounded in evidence.

The definitions of these elements are:
- **Content**: A description of the main subjects/concepts, objects, or actions depicted in the painting.
- **Context**: Background information about the historical, cultural, or biographical influences relevant to the painting.
- **Form**: An analysis of the artistic style and techniques used, including brushwork, color, composition, and use of light.

---CRITICAL GUIDELINES FOR METRIC OPTIMIZATION---
1. **Use exact terminology from retrieved context**: When the retrieved context mentions specific artist names, art movements, techniques, or historical periods, use those EXACT terms (e.g., "Pieter Bruegel the Elder", "Flemish Golden Age", "Dutch Golden Age", "Renaissance", "Post-Impressionism"). Do not paraphrase or use synonyms for these proper nouns.

2. **Mention specific entities and relationships**: Explicitly reference artist names, art movements, historical periods, and techniques mentioned in the retrieved context. This improves semantic proposition matching.

3. **Match ground truth style**: Use direct, factual statements similar to art catalog descriptions. Avoid flowery language, academic analysis, or verbose explanations.

4. **Prioritize key facts**: Focus on concrete details from the retrieved context rather than general observations. Include specific visual elements, techniques, or historical connections.

5. **Concise and factual**: Each section should be 1-2 sentences with concrete information. Total length: 20 words maximum.

---IMPORTANT FORMAT REQUIREMENTS---
- Structure your answer as **Content**, **Form**, and **Context** sections
- Each section should be 1-2 concise sentences (factual, not analytical)
- Total length: 25 words maximum
- Use descriptive, factual language similar to art catalog descriptions
- Include specific artist names, movements, and techniques when mentioned in retrieved context

######################
### Example 1:
######################

Retrieved Context:
- Entities: Pieter Bruegel the Elder (Artist), Flemish Golden Age (Historical Period), Harvest Scene (Painting)
- Relationships: Harvest Scene influenced by Bruegel's rural themes; belongs to Flemish Golden Age

Metadata:
Title: The Harvesters, Author: Unknown, Technique: Oil on wood, Timeframe: 1501-1600

Generation Plan:
[{{"step": 1, "goal": "Generate Content section", "output_format": "Content"}},
 {{"step": 2, "goal": "Generate Form section", "output_format": "Form"}}]

Generated description:
**Content**: The painting portrays a bustling harvest scene with farmers gathering wheat in expansive golden fields under a bright blue sky.
**Form**: The artist uses warm tones and detailed brushstrokes to create a realistic composition with natural lighting.

######################
### Example 2:
######################

Retrieved Context:
- Entities: Rachel Ruysch (Artist), Dutch Golden Age (Historical Period), Still Life with Flowers (Painting), Symbolism (Art Technique)
- Relationships: Still Life created by Ruysch; belongs to Dutch Golden Age; incorporates symbolic elements

Metadata:
Title: Still Life with Flowers, Author: Rachel Ruysch, Technique: Oil on panel, Timeframe: 1701-1750

Generation Plan:
[{{"step": 1, "goal": "Generate Content section", "output_format": "Content"}},
 {{"step": 2, "goal": "Generate Context section", "output_format": "Context"}}]

Generated description:
**Content**: The painting depicts an ornate arrangement of flowers in a glass vase, featuring roses, tulips, and carnations with some wilting petals.
**Context**: Created during the Dutch Golden Age, this still-life reflects the era's fascination with botanical accuracy and symbolic representation of life's transience.

######################
### Example 3 (Real SemArtv2 style):
######################

Retrieved Context:
- Entities: Master of Flémalle (Artist), Flemish Primitives (Art Movement), Annunciation (Painting), Robert Campin (Artist)
- Relationships: Annunciation attributed to Master of Flémalle; Master of Flémalle identified with Robert Campin; belongs to Flemish Primitives

Metadata:
Title: Annunciation, Author: Master of Flémalle, Technique: Tempera on oak, Timeframe: 1401-1450

Generation Plan:
[{{"step": 1, "goal": "Generate Content section", "output_format": "Content"}},
 {{"step": 2, "goal": "Generate Context section", "output_format": "Context"}}]

Generated description:
**Content**: The Virgin is seated in front of a low bench on the tiled floor, a sign of her humility. On Mary's lap is an open book, a second one lies on the table.
**Context**: This motif is possibly taken from devotional tracts of around 1400, which state that the Virgin was meditating on the Holy Scriptures when Gabriel entered.

######################
---Your Task---
######################

User query:
{query}

Metadata (if any):
{metadata}

Retrieved context:
{retrieved_context}

Generation plan:
{generation_plan}

IMPORTANT: 
- Extract and use EXACT artist names, art movements, techniques, and historical periods from the retrieved context
- Write factual, direct statements similar to the examples above
- Focus on concrete details rather than general observations
- Match the concise, factual style of SemArtv2 ground truth descriptions

Final answer:
"""

PROMPTS[
    "AGENTIC_FINAL_ANSWER_VQA"
] = """---Role---
Answer the user's question precisely and concisely about the artwork using the retrieved context and the generation plan.
Follow the plan steps in order and keep your answer grounded in evidence.

---CRITICAL GUIDELINES---
1. **Follow the generation plan**: Execute each step in the plan sequentially, using the specified evidence types.

2. **Ground in evidence**: Every claim must be supported by:
   - Visual evidence from the image
   - Information from the retrieved knowledge graph context
   - Metadata about the painting
   - Description text (if available)

3. **Multi-step reasoning**: Build your answer through logical reasoning steps, connecting visual observations 
   to historical context, artistic techniques, and symbolic meanings.

4. **Use exact terminology**: When the retrieved context mentions specific artist names, art movements, 
   techniques, or historical periods, use those EXACT terms (e.g., "Pieter Bruegel the Elder", 
   "Flemish Golden Age", "Renaissance"). Do not paraphrase proper nouns.

5. **Address the question directly**: Make sure your answer fully addresses what the question is asking. 
   If the question asks about relationships, symbolism, or comparisons, explicitly discuss those aspects.

6. **BE CONCISE AND PRECISE**: 
   - Keep your answer focused and to the point (aim for 100-200 words total, matching ground truth answer length)
   - Avoid verbose explanations, academic analysis, or unnecessary elaboration
   - Each reasoning step should be 1-2 sentences maximum
   - Skip ALL markdown formatting, headers, section breaks, bullet points, numbered lists, or structured sections
   - Write in a direct, factual style similar to art historical descriptions
   - Write as a SINGLE flowing paragraph that directly answers the question
   - Do NOT include phrases like "I will work through", "Let me analyze", "Step 1/2/3", or any meta-commentary
   - Start directly with the answer content
   - Be factual and direct, similar to museum catalog descriptions

---EXAMPLE OF GOOD CONCISE ANSWER---
Question: "How does this painting reconcile the saint's spiritual identity with his representation as a historical figure of worldly power?"

Good Answer (concise, ~150 words):
The painting portrays St. Ladislaus without a halo, marking him as a historical personage rather than a purely spiritual figure, which indicates it was intended for a secular building. However, his dual identity as both Christian king and Christian knight is established through the visual composition. He is shown wearing royal regalia—a crown and an ample, richly embroidered cloak studded with pearls—asserting his worldly authority. The presence of the national emblem on the voluted shield in the foreground reinforces his role as a ruler of temporal power. Yet the background scenes depicting two episodes from the king's life serve to identify him as embodying the ideal unity of Christian kingship and knighthood. The Renaissance palace interior setting further situates him as a historical ruler. Thus, the painting balances secular authority with spiritual ideals through compositional choices that emphasize both royal symbols and legendary narrative.

---Your Task---
User question:
{query}

Metadata (if any):
{metadata}

Retrieved context (knowledge graph subgraph):
{retrieved_context}

Generation plan (reasoning steps to follow):
{generation_plan}

Answer the question by following the generation plan step by step. For each step in the plan:
- Use the specified evidence types (visual, metadata, kg, text)
- Ground your reasoning in the retrieved context
- Build logically from one step to the next
- Connect visual observations to historical/artistic context

Provide a clear, concise answer (100-200 words) that demonstrates multi-step reasoning and is fully grounded in the evidence. Write as a SINGLE flowing paragraph without any markdown formatting, headers, or structure.

Final answer:
"""