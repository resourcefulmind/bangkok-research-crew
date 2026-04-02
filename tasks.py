from crewai import Task
from agents import (
    search_agent, 
    novelty_evaluator, 
    impact_evaluator, 
    practical_evaluator, 
    ranking_synthesizer, 
    output_agent, 
)

# Task 1 - Search Arxiv for papers
search_task = Task(       
    description=(                                                                                                                                                                                                                             
        "Search ArXiv for all research papers published on {date} in the "
        "following categories: {categories}. For each paper found, collect: "                                                                                                                                                                 
        "the title, full list of authors, complete abstract, ArXiv URL, "
        "direct PDF link, and all categories the paper is listed under. "                                                                                                                                                                     
        "Return the complete list of papers with all metadata. Do not "
        "evaluate or filter the papers — return everything you find."                                                                                                                                                                         
    ),                                 
    expected_output=(                                                                                                                                                                                                                         
        "A complete list of all papers found, where each paper includes: "
        "title, authors, abstract, ArXiv URL, PDF link, and categories. "                                                                                                                                                                     
        "Also include the total number of papers found."
    ),                                                                                                                                                                                                                                        
    agent=search_agent,                                                                                                                                                                                                                       
    human_input=True,                                                                                                                                                                                                                         
)

# Task 2 - Evaluate Novelty of Papers
novelty_task = Task(                                                                                                                                                                                                                          
    description=(       
        "Evaluate each paper from the search results on its NOVELTY only. "                                                                                                                                                                   
        "For each paper, assess: Has this specific approach been done before? "                                                                                                                                                               
        "If similar work exists, how is this fundamentally different? Does it "                                                                                                                                                               
        "challenge an existing assumption in the field? Could other researchers "                                                                                                                                                             
        "build on this in surprising ways? Assign a novelty score from 1 to 10 "
        "where 1 is purely incremental and 10 is a groundbreaking new idea. "                                                                                                                                                                 
        "Provide 1-2 sentences of reasoning for each score."
    ),                                                                                                                                                                                                                                        
    expected_output=(                  
        "A list of all papers with their novelty scores (1-10) and short "                                                                                                                                                                    
        "reasoning for each score. Group papers into tiers: Breakthrough "                                                                                                                                                                    
        "(8-10), Significant (5-7), Incremental (1-4)."                                                                                                                                                                                       
    ),                                                                                                                                                                                                                                        
    agent=novelty_evaluator,                                                                                                                                                                                                                  
    context=[search_task],                                                                                                                                                                                                                    
)

# Task 3 - Evaluate Impact of Papers
impact_task = Task(                                                                                                                                                                                                                           
    description=(                      
        "Evaluate each paper from the search results on its IMPACT and "                                                                                                                                                                      
        "RIGOR only. For each paper, assess these four dimensions: SCALE — "
        "how many datasets, tasks, or domains were evaluated? RIGOR — is "                                                                                                                                                                    
        "there mention of ablation studies, error bars, or statistical "                                                                                                                                                                      
        "significance? REPRODUCIBILITY — is code or data availability "                                                                                                                                                                       
        "mentioned? Are implementation details clear? BENCHMARKS — does this "                                                                                                                                                                
        "paper set a new standard or introduce a new evaluation method? "                                                                                                                                                                     
        "Assign an impact score from 1 to 10 where 1 is minimal validation "                                                                                                                                                                  
        "and 10 is comprehensive, field-defining work. Provide 1-2 sentences "                                                                                                                                                                
        "of reasoning for each score."                                                                                                                                                                                                        
    ),                                                                                                                                                                                                                                        
    expected_output=(                                                                                                                                                                                                                         
        "A list of all papers with their impact scores (1-10) and short "                                                                                                                                                                     
        "reasoning for each score. Group papers into tiers: Field-defining "                                                                                                                                                                  
        "(8-10), Solid validation (5-7), Limited validation (1-4)."                                                                                                                                                                           
    ),                                                                                                                                                                                                                                        
    agent=impact_evaluator,                                                                                                                                                                                                                   
    context=[search_task],                                                                                                                                                                                                                    
)

# Task 4 - Evaluate Practical Relevance of Papers
practical_task = Task(                                                                                                                                                                                                                        
    description=(                      
        "Evaluate each paper from the search results on its PRACTICAL "
        "APPLICABILITY only. For each paper, assess these five dimensions: "
        "PROBLEM RELEVANCE — does this solve a real problem practitioners "                                                                                                                                                                   
        "care about? DEPLOYMENT READINESS — could someone implement this in "
        "a real system? EFFICIENCY — does it reduce compute, memory, or "                                                                                                                                                                     
        "time requirements? ECOSYSTEM FIT — does it work with existing tools "
        "and frameworks? GENERALIZATION — can it be applied beyond the "                                                                                                                                                                      
        "specific task shown in the paper? Assign a practicality score from "
        "1 to 10 where 1 is purely academic and 10 is immediately deployable. "                                                                                                                                                               
        "Provide 1-2 sentences of reasoning for each score."
    ),                                                                                                                                                                                                                                        
    expected_output=(                                                                                                                                                                                                                         
        "A list of all papers with their practicality scores (1-10) and "                                                                                                                                                                     
        "short reasoning for each score. Group papers into tiers: Immediately "                                                                                                                                                               
        "useful (8-10), Moderately applicable (5-7), Primarily academic (1-4)."                                                                                                                                                               
    ),                                                                                                                                                                                                                                        
    agent=practical_evaluator,                                                                                                                                                                                                                
    context=[search_task],                                                                                                                                                                                                                    
)

# Task 5 - Rank the Papers
ranking_task = Task(      
    description=(                                                                                                                                                                                                                             
        "Using the novelty, impact, and practicality evaluations from the "
        "three specialist evaluators, produce a final ranking of the top 10 "                                                                                                                                                                 
        "most important papers. For each paper, calculate a composite score "
        "by weighing novelty, impact, and practicality roughly equally. When "                                                                                                                                                                
        "evaluators disagree — for example, high novelty but low practicality "
        "— explain the trade-off and justify your ranking decision. For each "                                                                                                                                                                
        "paper in the top 10, provide: the final rank (1-10), the composite "
        "score out of 10, the individual novelty score, impact score, and "
        "practicality score, and a 2-3 sentence explanation of why it earned "
        "its position."                                                                                                                                                                                                                       
    ),                                                                                                                                                                                                                                        
    expected_output=(                                                                                                                                                                                                                         
        "A ranked list of the top 10 papers. Each entry includes: rank "                                                                                                                                                                      
        "position, paper title, authors, ArXiv URL, PDF link, composite "                                                                                                                                                                     
        "score (out of 10), individual novelty/impact/practicality scores, "                                                                                                                                                                  
        "and a clear explanation of the ranking rationale."                                                                                                                                                                                   
    ),                                 
    agent=ranking_synthesizer,                                                                                                                                                                                                                
    context=[novelty_task, impact_task, practical_task],
    human_input=True,                                                                                                                                                                                                                         
)

# Task 6 - Generate Report
output_task = Task(     
    description=(         
        "Generate a self-contained HTML report for the top 10 ranked research "                                                                                                                                                               
        "papers. The report must use inline CSS with no external dependencies "
        "and no JavaScript libraries. Design it with an Apple-inspired "                                                                                                                                                                      
        "glassmorphism aesthetic: frosted glass card effects using "                                                                                                                                                                          
        "backdrop-filter blur and rgba backgrounds, layered box-shadows for "                                                                                                                                                                 
        "depth, soft rounded corners with 16px border-radius, a soft gradient "                                                                                                                                                               
        "background, and smooth hover transitions. Use the system font stack "
        "(-apple-system, BlinkMacSystemFont, Segoe UI, Roboto). The layout "
        "should be a centered container with max-width 900px. Include a hero "                                                                                                                                                                
        "header section showing the date, total papers found, and categories "
        "searched. Each paper is a glassmorphic card showing: rank badge, "                                                                                                                                                                   
        "composite score, title linked to the ArXiv page, authors, a one-line "                                                                                                                                                               
        "summary of why the paper matters, novelty/impact/practicality scores "                                                                                                                                                               
        "as color-coded pill badges, the abstract in a collapsible details "                                                                                                                                                                  
        "element, category tags, and a PDF link button. Include a footer with "                                                                                                                                                               
        "search parameters and generation timestamp. Support dark mode via "                                                                                                                                                                  
        "prefers-color-scheme media query and print styles via print media "                                                                                                                                                                  
        "query. Save the file as 'output/report.html'."                                                                                                                                                                                       
    ),                                                                                                                                                                                                                                        
    expected_output=(                  
        "A complete, self-contained HTML file with inline CSS. The file "                                                                                                                                                                     
        "should be visually polished with glassmorphism design, fully "
        "responsive, and saved to output/report.html."                                                                                                                                                                                        
    ),                                                                                                                                                                                                                                        
    agent=output_agent,                                                                                                                                                                                                                       
    context=[ranking_task],                                                                                                                                                                                                                   
    output_file="output/report.html",                                                                                                                                                                                                         
)