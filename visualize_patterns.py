#!/usr/bin/env python
"""
Script to visualize all trading patterns with their shortened breakout/breakdown phases
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from test_minute_timerag_qqq import create_patterns

# Create patterns directory if it doesn't exist
patterns_dir = "patterns"
if not os.path.exists(patterns_dir):
    os.makedirs(patterns_dir)
    print(f"Created directory: {patterns_dir}")

# Get the patterns from the main script
patterns = create_patterns()

# Define pattern-specific settings
pattern_info = {
    "ascending_triangle": {
        "title": "Ascending Triangle Pattern",
        "breakout_start": 52,
        "breakout_label": "Breakout (8 periods)",
        "features": [
            {"type": "hline", "y": 0.7, "color": "r", "style": "--", "label": "Resistance"},
            {"type": "line", "points": [(15, 0.435), (35, 0.53)], "color": "g", "style": "--", "label": "Rising Support"}
        ]
    },
    "bull_flag": {
        "title": "Bull Flag Pattern with 4 Oscillations",
        "breakout_start": 52,
        "breakout_label": "Breakout (8 periods)",
        "features": [
            {"type": "span", "start": 0, "end": 20, "color": "green", "alpha": 0.2, "label": "Flag Pole"},
            {"type": "span", "start": 20, "end": 52, "color": "yellow", "alpha": 0.2, "label": "Flag Consolidation"}
        ]
    },
    "double_bottom": {
        "title": "Double Bottom Pattern",
        "breakout_start": 52,
        "breakout_label": "Breakout (8 periods)",
        "features": [
            {"type": "point", "x": 15, "y": 0.1, "label": "First Bottom"},
            {"type": "point", "x": 40, "y": 0.1, "label": "Second Bottom"}
        ]
    },
    "head_and_shoulders": {
        "title": "Head and Shoulders Pattern",
        "breakout_start": 52,
        "breakout_label": "Breakdown (8 periods)",
        "features": [
            {"type": "point", "x": 7, "y": 0.5, "label": "Left Shoulder"},
            {"type": "point", "x": 22, "y": 0.7, "label": "Head"},
            {"type": "point", "x": 37, "y": 0.5, "label": "Right Shoulder"},
            {"type": "hline", "y": 0.3, "color": "r", "style": "--", "label": "Neckline"}
        ]
    },
    "cup_and_handle": {
        "title": "Cup and Handle Pattern",
        "breakout_start": 52,
        "breakout_label": "Breakout (8 periods)",
        "features": [
            {"type": "span", "start": 0, "end": 30, "color": "cyan", "alpha": 0.2, "label": "Cup"},
            {"type": "span", "start": 30, "end": 52, "color": "yellow", "alpha": 0.2, "label": "Handle"}
        ]
    }
}

# Create a figure with subplots for each pattern
fig, axes = plt.subplots(len(patterns), 1, figsize=(12, 4 * len(patterns)))

# Plot each pattern
for i, (name, pattern) in enumerate(patterns.items()):
    ax = axes[i]
    
    # Plot the pattern
    ax.plot(np.arange(len(pattern)), pattern, 'b-', linewidth=2, label='Price')
    
    # Highlight the breakout/breakdown area
    info = pattern_info[name]
    breakout_start = info["breakout_start"]
    ax.axvspan(breakout_start, 60, color='red', alpha=0.2, label=info["breakout_label"])
    
    # Add pattern-specific features
    for feature in info["features"]:
        if feature["type"] == "hline":
            ax.axhline(y=feature["y"], color=feature["color"], linestyle=feature["style"], label=feature["label"])
        elif feature["type"] == "line":
            points = feature["points"]
            ax.plot([p[0] for p in points], [p[1] for p in points], 
                   color=feature["color"], linestyle=feature["style"], label=feature["label"])
        elif feature["type"] == "span":
            ax.axvspan(feature["start"], feature["end"], 
                      color=feature["color"], alpha=feature["alpha"], label=feature["label"])
        elif feature["type"] == "point":
            ax.plot(feature["x"], feature["y"], 'o', color='red', markersize=5)
            ax.annotate(feature["label"], 
                       xy=(feature["x"], feature["y"]),
                       xytext=(feature["x"]+2, feature["y"] - 0.1),
                       arrowprops=dict(facecolor='black', shrink=0.05, width=1))
    
    # Annotate the breakout/breakdown
    breakout_y = pattern[breakout_start]
    ax.annotate(info["breakout_label"], 
               xy=(breakout_start + 4, breakout_y),
               xytext=(breakout_start + 4, breakout_y + 0.15),
               arrowprops=dict(facecolor='black', shrink=0.05, width=1))
    
    # Add labels and title
    ax.set_xlabel('Time Periods')
    ax.set_ylabel('Price')
    ax.set_title(info["title"])
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{patterns_dir}/all_patterns.png', dpi=150)
print(f"Saved visualization to {patterns_dir}/all_patterns.png")

# Show the plot
plt.show()

# Additionally, create individual high-resolution images for each pattern
for name, pattern in patterns.items():
    info = pattern_info[name]
    plt.figure(figsize=(12, 6))
    
    # Plot the pattern
    plt.plot(np.arange(len(pattern)), pattern, 'b-', linewidth=2, label='Price')
    
    # Highlight the breakout/breakdown area
    breakout_start = info["breakout_start"]
    plt.axvspan(breakout_start, 60, color='red', alpha=0.2, label=info["breakout_label"])
    
    # Add pattern-specific features
    for feature in info["features"]:
        if feature["type"] == "hline":
            plt.axhline(y=feature["y"], color=feature["color"], linestyle=feature["style"], label=feature["label"])
        elif feature["type"] == "line":
            points = feature["points"]
            plt.plot([p[0] for p in points], [p[1] for p in points], 
                    color=feature["color"], linestyle=feature["style"], label=feature["label"])
        elif feature["type"] == "span":
            plt.axvspan(feature["start"], feature["end"], 
                       color=feature["color"], alpha=feature["alpha"], label=feature["label"])
        elif feature["type"] == "point":
            plt.plot(feature["x"], feature["y"], 'o', color='red', markersize=5)
            plt.annotate(feature["label"], 
                        xy=(feature["x"], feature["y"]),
                        xytext=(feature["x"]+2, feature["y"] - 0.1),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1))
    
    # Annotate the breakout/breakdown
    breakout_y = pattern[breakout_start]
    plt.annotate(info["breakout_label"], 
                xy=(breakout_start + 4, breakout_y),
                xytext=(breakout_start + 4, breakout_y + 0.15),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1))
    
    # Add labels and title
    plt.xlabel('Time Periods')
    plt.ylabel('Price')
    plt.title(info["title"])
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Save individual pattern
    filename = f"{patterns_dir}/{name}_pattern.png"
    plt.savefig(filename, dpi=150)
    print(f"Saved {filename}")
    plt.close() 