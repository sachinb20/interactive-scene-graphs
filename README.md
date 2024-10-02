# Interactive Scene Graphs

## Motivation
Current Scene-Graph Representations don’t allow for quick scene changes in cases of long-term planning.

## Contribution
- **Real-time Scene Representation Update:** Method to update scene representations in real time due to changes from robots or external agents.
- **LLM-based Planning and Replanning:** Leveraging Large Language Models (LLMs) for planning and replanning when external changes occur in the environment.

## Method

### Exploration
- **Pre-defined Landmarks:** Utilize predefined points of interest.
- **Object Detection with Detic:** Objects detected using Detic model.
- **Receptacle State Features with CLIP:** Use CLIP to analyze and identify the state of receptacles.
![Your Image Alt Text](./exploration.png)
### LLM-Replanning
During task execution, if the scene changes due to an external agent, feedback is provided to the LLM agent for replanning. This feedback signal helps adapt the task dynamically.

### Task Replanning with Scene Representation

Let:
- \( T \) be the set of tasks.
- \( A \) be the action history.
- \( S_{scene} \) be the current scene representation.
- \( S_{physical} \) be the physical state of the scene (detected by sensors or agents).
- \( F \) be the feedback provided to the system.
- \( T_{updated} \) be the set of updated tasks.

We want to update the tasks only if the scene representation is different from the physical scene. This can be formalized as:

\[
T_{updated} = \\begin{cases}
\text{Replan}(T, A, F) & \text{if } S_{scene} \neq S_{physical} \\\\
T & \text{otherwise}
\end{cases}
\]

Where:
- \( \text{Replan}(T, A, F) \) is a function that replans the task based on the current task \( T \), the action history \( A \), and the feedback \( F \).
- \( S_{scene} \neq S_{physical} \) indicates a discrepancy between the scene representation and the actual physical state, triggering the need for task updates.

## Feedback Signals
We use two feedback signals:
1. **Visual Feedback**
2. **Physical Feedback**

## Experiments
We conducted the following experiments:

### Multi-Step Commands
To demonstrate the effectiveness of the updatable scene representation.

#### Example: Object Displacement
