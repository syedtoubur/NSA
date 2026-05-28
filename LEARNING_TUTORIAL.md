# NSA ARC Solver - Step-by-Step Tutorial

## STEP 1: Foundation Assessment

### Your Performance So Far

**Code Implementation:** ✅ EXCELLENT
- `get_grid_properties()`: Perfect! You understood numpy array shapes correctly
- Color extraction: Correct use of `np.unique()`

**MCQ Answers:**
1. ✅ **CORRECT** - B) Top-left corner stays 1, rest becomes 0
2. ✅ **CORRECT** - B) To group related pixels into objects/concepts
3. ✅ **CORRECT** - A) Pixels that are adjacent and have the same color
4. ❓ **Let's explore this together**

---

## Explanation: Question 4 (Abstraction Strategies)

### What Does "Abstraction Strategy" Mean?

In this codebase, there are **6 different ways** to convert a grid into a graph:

```
Abstractions Available:
1. nbccg - Neighbor Connected Color Grid (treats black as boundaries)
2. ccgbr - Connected Color Grid with Border Recognition  
3. nbvcg - Neighbor Vertical Color Grid
4. nbhcg - Neighbor Horizontal Color Grid
5. ccg   - Simple Connected Color Grid
6. mcccg - Multi-Color Connected Color Grid
7. na    - Noise Agnostic (ignores background)
8. lrg   - Line Recognition Grid
```

Each produces a different **abstract graph** from the same input grid.

**Example:** If your input is:
```
[1, 0, 2]
[0, 1, 0]
[2, 0, 1]
```

**Abstraction 1 (nbccg)** might see:
- Object A: The 1s (if connected)
- Object B: The 2s (if connected)
- Background: The 0s

**Abstraction 2 (na)** might see:
- Completely different grouping (ignores 0s)

### Answer to Question 4:

**Correct Answer: C) 6+ (multiple abstraction strategies)**

**Why?**
The algorithm tries **multiple abstractions in parallel**:
- It doesn't know which abstraction will work best for YOUR task
- So it tries all of them and explores each one
- If one abstraction finds a solution faster, it uses that
- The system is designed to be **abstraction-agnostic**

---

## Now Let's Fix the BFS/DFS Problem

### Understanding BFS (Breadth-First Search)

BFS finds all pixels connected to a starting pixel:

```
Grid:           Pixel (0,0) connects to:
[1, 1, 0]       → (0,1) - same color, adjacent
[1, 0, 0]       → (1,0) - same color, adjacent
[0, 0, 0]       
                Then from (0,1):
                → (0,0) - already visited
                → No new neighbors
                
                Then from (1,0):
                → (0,0) - already visited
                → No new neighbors
                
                Result: One connected component of 1s
```

### BFS Implementation:

```python
from collections import deque

def count_connected_components(grid):
    """Count objects of the same color using BFS"""
    grid_array = np.array(grid)
    visited = set()
    components = 0
    
    # For each pixel in the grid
    for i in range(grid_array.shape[0]):
        for j in range(grid_array.shape[1]):
            # If not visited and not background (color 0)
            if (i, j) not in visited and grid_array[i, j] != 0:
                # This is a new object! Count it
                components += 1
                
                # BFS to find all connected pixels
                queue = deque([(i, j)])
                visited.add((i, j))
                
                while queue:
                    current_i, current_j = queue.popleft()
                    current_color = grid_array[current_i, current_j]
                    
                    # Check all 4 neighbors (up, down, left, right)
                    neighbors = [
                        (current_i - 1, current_j),  # Up
                        (current_i + 1, current_j),  # Down
                        (current_i, current_j - 1),  # Left
                        (current_i, current_j + 1)   # Right
                    ]
                    
                    for ni, nj in neighbors:
                        # Check if neighbor is in bounds
                        if 0 <= ni < grid_array.shape[0] and 0 <= nj < grid_array.shape[1]:
                            # Check if not visited and same color
                            if (ni, nj) not in visited and grid_array[ni, nj] == current_color:
                                visited.add((ni, nj))
                                queue.append((ni, nj))
    
    return components
```

### Step-by-Step Breakdown:

1. **deque** = Double-ended queue (efficient for BFS)
2. **visited.add((i, j))** = Mark pixel as visited
3. **queue.popleft()** = Get the first pixel from queue
4. **neighbors list** = All 4 adjacent cells
5. **Bounds check** = Make sure we're still in the grid
6. **Color check** = Only add if same color and not visited
7. **queue.append()** = Add to queue for processing

---

## Complete Exercise 1 Solution

```python
# ============================================================================
# EXERCISE 1: COMPLETE SOLUTION
# ============================================================================

from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ARC uses 10 colors (0-9)
ARC_COLORS = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"
]

def get_grid_properties(grid):
    """Extract basic properties of a grid"""
    grid_array = np.array(grid)
    
    height = grid_array.shape[0]
    width = grid_array.shape[1]
    unique_colors = np.unique(grid_array)
    
    return {
        "height": height,
        "width": width,
        "unique_colors": list(unique_colors),
        "size": height * width
    }

def count_connected_components(grid):
    """Count objects of the same color using BFS"""
    grid_array = np.array(grid)
    visited = set()
    components = 0
    
    for i in range(grid_array.shape[0]):
        for j in range(grid_array.shape[1]):
            if (i, j) not in visited and grid_array[i, j] != 0:
                components += 1
                
                # BFS
                queue = deque([(i, j)])
                visited.add((i, j))
                
                while queue:
                    current_i, current_j = queue.popleft()
                    current_color = grid_array[current_i, current_j]
                    
                    neighbors = [
                        (current_i - 1, current_j),
                        (current_i + 1, current_j),
                        (current_i, current_j - 1),
                        (current_i, current_j + 1)
                    ]
                    
                    for ni, nj in neighbors:
                        if (0 <= ni < grid_array.shape[0] and 
                            0 <= nj < grid_array.shape[1] and
                            (ni, nj) not in visited and 
                            grid_array[ni, nj] == current_color):
                            visited.add((ni, nj))
                            queue.append((ni, nj))
    
    return components

def visualize_grid(grid, title="Grid"):
    """Visualize a single grid"""
    cmap = ListedColormap(ARC_COLORS)
    plt.figure(figsize=(4, 4))
    plt.imshow(np.array(grid), cmap=cmap, vmin=0, vmax=9)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks([])
    plt.yticks([])
    plt.grid(True, which='both', color='lightgray', linewidth=0.5)
    plt.show()

# Sample ARC Task
sample_arc_task = {
    "train": [
        {
            "input": [[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]],
            "output": [[1, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]]
        }
    ],
    "test": [
        {
            "input": [[2, 2, 2],
                      [2, 2, 2],
                      [2, 2, 2]],
            "output": []
        }
    ]
}

# Test the functions
print("=" * 50)
print("EXERCISE 1: COMPLETE SOLUTION")
print("=" * 50)

input_grid = sample_arc_task["train"][0]["input"]
output_grid = sample_arc_task["train"][0]["output"]

print("\n[INPUT GRID PROPERTIES]")
print(get_grid_properties(input_grid))

print("\n[OUTPUT GRID PROPERTIES]")
print(get_grid_properties(output_grid))

print("\n[CONNECTED COMPONENTS]")
print(f"Input components: {count_connected_components(input_grid)}")
print(f"Output components: {count_connected_components(output_grid)}")

print("\n[TRANSFORMATION ANALYSIS]")
print("Input: One object (all 1s connected in a 3x3 block)")
print("Output: One object (only top-left 1 remains, rest become 0s)")
print("RULE: Keep top-left pixel, change rest to background color (0)")
```

---

## Key Concepts You've Learned

### 1. Grid Properties
- **Height/Width**: Using `.shape[0]` and `.shape[1]`
- **Unique Colors**: Using `np.unique()`
- **Size**: Total pixels in grid

### 2. Connected Components
- **Definition**: Pixels of the same color that are adjacent
- **Finding them**: BFS/DFS algorithm
- **Why it matters**: Each component = one abstract "object"

### 3. BFS Algorithm
```
Queue Operations:
- .append()  → Add to back
- .popleft() → Remove from front

Visited Set:
- Prevents processing same pixel twice
- Makes algorithm efficient

Neighbor Checking:
- 4-connected: up, down, left, right
- 8-connected: +4 diagonals (not used here)
```

---

## Summary of Answers

| Question | Your Answer | Correct Answer | Status |
|----------|------------|---------------|---------|
| 1 | B | B | ✅ |
| 2 | B | B | ✅ |
| 3 | A | A | ✅ |
| 4 | ? | C (6+) | 📚 See above |

---

## Next Steps

You're ready for **STEP 2**! We'll:
1. Load real ARC datasets
2. Create actual Image objects
3. Explore different abstractions
4. See how graphs are built from grids

**Are you ready?** Let me know once you've:
- ✅ Understood the BFS algorithm
- ✅ Run the complete solution
- ✅ Tested with a few different grids
- ✅ Confirmed your understanding
