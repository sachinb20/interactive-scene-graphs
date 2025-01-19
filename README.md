# **Interactive Scene Graphs**  

Current scene-graph representations often struggle to adapt to dynamic environments, especially during long-term planning tasks. This repository provides tools and methodologies to enhance scene-graph representations for real-time adaptability, enabling more efficient planning and replanning in response to environmental changes.  

## **Key Contributions**  

1. **Real-Time Scene-Representation Updates**  
   - Developed a robust method to dynamically update scene representations in response to changes caused by robots or external agents.  
   - Ensures accurate alignment between the internal scene representation and the physical environment for seamless task execution.  

2. **LLM-Based Planning and Replanning**  
   - Integrates large language models (LLMs) for task planning and intelligent replanning.  
   - Adapts to external changes in the environment by recalculating plans based on updated scene representations, action history, and feedback.  
 

## **Algorithm**  

#### Input:  
- \( T \): Set of tasks.  
- \( A_{\text{history}} \): Action history.  
- \( S_{\text{scene}} \): Current scene representation.  
- \( S_{\text{physical}} \): Physical state of the scene (from sensors or agents).  
- \( F \): Feedback provided to the system.  

#### Output:  
- Updated \( S_{\text{scene}} \): Scene representation.  
- \( A, \text{Object, Target} \): Tuple representing action primitives, objects, and targets after replanning (if necessary).  

---

#### Algorithm:  

1. **Initial Planning**:  
   Call the function \( \text{Plan}(T, S_{\text{scene}}) \):  
   - Input: Tasks \( T \) and current scene representation \( S_{\text{scene}} \).  
   - Output: Tuple \( (A, \text{Object, Target}) \):  
     - \( A \): Action primitives.  
     - \( \text{Object} \): Objects involved in the plan.  
     - \( \text{Target} \): Target locations or states.  

2. **Scene Check**:  
   Compare \( S_{\text{scene}} \) with \( S_{\text{physical}} \):  
   - If \( S_{\text{scene}} \neq S_{\text{physical}} \):  
     - Update \( S_{\text{scene}} \) to \( S_{\text{physical}} \).  
     - Proceed to replanning.  

3. **Replanning**:  
   Call \( \text{Replan}(T, A_{\text{history}}, F, S_{\text{scene}}) \):  
   - Input: Current tasks \( T \), action history \( A_{\text{history}} \), feedback \( F \), and updated \( S_{\text{scene}} \).  
   - Output: Tuple \( (A, \text{Object, Target}) \).  


---


## **Getting Started**  

### Prerequisites  

- Python 3.8+  
- Required Libraries: Ai2Thor, DETIC, openai

## **Usage**  

### Running Experiments  

To run experiments, use the `run_experiment.py` script with configurable parameters. Below are the details for each argument:  

| **Argument**               | **Default**                                                                                     | **Description**                                                                 |
|----------------------------|-------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|                                           |
| `--experiment_type`        | `"multi_step"`                                                                                 | Type of experiment: `multi_step`, `obj_displaced`, or `obj_introduced`.         |
| `--temperature_list`       | `["0.3", "0.6", "0.9"]`                                                                        | List of temperature values for the LLM.                                        |
| `--scene`                  | `"FloorPlan4"`                                                                                | Name of the scene to use in the experiment.                                     |
| `--visual_feedback_limit`  | `1`                                                                                           | Number of visual feedback steps allowed.                                       |
| `--action_feedback_limit`  | `2`                                                                                           | Number of action feedback steps allowed.                                       |
| `--embodiment`             | `"Bi-manipulation"`                                                                           | Type of embodiment: e.g., `Bi-manipulation`, `Single-arm`, etc.                 |
