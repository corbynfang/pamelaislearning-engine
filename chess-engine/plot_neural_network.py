import os
import subprocess
import shutil

def plot_chess_neural_network_3d():
    working_dir = os.path.abspath(os.path.curdir)
    print(f"Working directory: {working_dir}")

    # Create a 3D LaTeX visualization of the neural network
    latex_content = r"""
\documentclass[border=8pt, tikz]{standalone}
\usepackage{tikz}
\usetikzlibrary{positioning,3d,shapes.geometric}
\usepackage{amsmath}

\begin{document}
\def\ConvColor{rgb:yellow,5;red,2.5;white,5}
\def\ConvReluColor{rgb:yellow,5;red,5;white,5}
\def\PoolColor{rgb:red,1;black,0.3}
\def\FcColor{rgb:blue,5;red,2.5;white,5}
\def\SoftmaxColor{rgb:magenta,5;black,7}
\def\ResidualColor{rgb:green,5;red,1;white,2}
\def\BatchNormColor{rgb:blue,2;green,5;white,5}

\begin{tikzpicture}[
    x={(0.866cm,-0.5cm)},
    y={(0cm,1cm)},
    z={(-0.866cm,-0.5cm)},
    scale=0.7,
    every node/.style={font=\footnotesize}
]
    % Define cube function to draw 3D layers
    \def\cube[#1]#2#3#4#5{
        \def\x{#2}
        \def\y{#3}
        \def\z{#4}
        \def\col{#5}

        \coordinate (A) at (\x,\y,\z);
        \coordinate (B) at (\x+3,\y,\z);
        \coordinate (C) at (\x+3,\y+\cubey,\z);
        \coordinate (D) at (\x,\y+\cubey,\z);
        \coordinate (E) at (\x,\y,\z+\cubez);
        \coordinate (F) at (\x+3,\y,\z+\cubez);
        \coordinate (G) at (\x+3,\y+\cubey,\z+\cubez);
        \coordinate (H) at (\x,\y+\cubey,\z+\cubez);

        % Draw the cube faces with opacity
        \filldraw[fill=\col, fill opacity=0.7, draw=black, #1]
            (D) -- (C) -- (G) -- (H) -- cycle; % front
        \filldraw[fill=\col, fill opacity=0.7, draw=black, #1]
            (B) -- (C) -- (G) -- (F) -- cycle; % right
        \filldraw[fill=\col, fill opacity=0.7, draw=black, #1]
            (A) -- (B) -- (F) -- (E) -- cycle; % bottom
        \filldraw[fill=\col, fill opacity=0.4, draw=black, #1]
            (A) -- (D) -- (H) -- (E) -- cycle; % left
        \filldraw[fill=\col, fill opacity=0.5, draw=black, #1]
            (E) -- (F) -- (G) -- (H) -- cycle; % top
        \filldraw[fill=\col, fill opacity=0.3, draw=black, #1]
            (A) -- (B) -- (C) -- (D) -- cycle; % back
    }

    % Layer dimensions
    \def\cubey{5}  % Width
    \def\cubez{5}  % Depth (for 3D effect)

    % Input layer
    \cube[]{0}{0}{0}{\ConvColor}
    \node at (1.5,2.5,5.5) {Input: $19 \times 8 \times 8$};

    % Initial convolution with BatchNorm + ReLU
    \cube[]{0}{-7}{0}{\ConvReluColor}
    \node at (1.5,-4.5,5.5) {Initial Conv $3 \times 3 \times 128$};
    \node at (1.5,-3.5,5.5) {BatchNorm + ReLU};

    % Draw arrow from input to initial conv
    \draw[->, thick] (1.5,-0.5,2.5) -- (1.5,-6.5,2.5);

    % Residual Block 1
    \cube[]{0}{-14}{0}{\ResidualColor}
    \node at (1.5,-11.5,5.5) {Residual Block 1};
    \draw[->, thick] (1.5,-7.5,2.5) -- (1.5,-13.5,2.5);

    % Residual Block 2
    \cube[]{0}{-21}{0}{\ResidualColor}
    \node at (1.5,-18.5,5.5) {Residual Block 2};
    \draw[->, thick] (1.5,-14.5,2.5) -- (1.5,-20.5,2.5);

    % ... (dots for remaining blocks)
    \node at (1.5,-24,2.5) {$\vdots$};

    % Residual Block 10
    \cube[]{0}{-28}{0}{\ResidualColor}
    \node at (1.5,-25.5,5.5) {Residual Block 10};
    \draw[->, thick] (1.5,-24.5,2.5) -- (1.5,-27.5,2.5);

    % Fork point
    \node[draw, ellipse, minimum width=4cm, minimum height=1.5cm] at (1.5,-31.5,2.5) {Feature Map: $128 \times 8 \times 8$};
    \draw[->, thick] (1.5,-28.5,2.5) -- (1.5,-30.7,2.5);

    % Value Head (left branch)
    \def\cubey{3}  % Make value head cubes smaller
    \def\cubez{3}

    % Value Conv + BatchNorm + ReLU
    \cube[]{-6}{-36}{0}{\ConvColor}
    \node at (-4.5,-34.5,3.5) {Value Conv $1 \times 1 \times 32$};
    \node at (-4.5,-33.5,3.5) {BatchNorm + ReLU};

    % Flatten layer
    \cube[]{-6}{-42}{0}{white}
    \node at (-4.5,-40.5,3.5) {Flatten: $32 \times 8 \times 8 = 2048$};

    % FC layers
    \cube[]{-6}{-48}{0}{\FcColor}
    \node at (-4.5,-46.5,3.5) {FC-256 + ReLU};

    \cube[]{-6}{-54}{0}{\FcColor}
    \node at (-4.5,-52.5,3.5) {FC-128 + ReLU};

    \cube[]{-6}{-60}{0}{\FcColor}
    \node at (-4.5,-58.5,3.5) {FC-1};

    % Tanh activation
    \cube[]{-6}{-66}{0}{\SoftmaxColor}
    \node at (-4.5,-64.5,3.5) {tanh};

    % Connect Value Head
    \draw[->, thick] (0,-31.5,2.5) -- (-4.5,-35.5,1.5);
    \draw[->, thick] (-4.5,-36.5,1.5) -- (-4.5,-41.5,1.5);
    \draw[->, thick] (-4.5,-42.5,1.5) -- (-4.5,-47.5,1.5);
    \draw[->, thick] (-4.5,-48.5,1.5) -- (-4.5,-53.5,1.5);
    \draw[->, thick] (-4.5,-54.5,1.5) -- (-4.5,-59.5,1.5);
    \draw[->, thick] (-4.5,-60.5,1.5) -- (-4.5,-65.5,1.5);

    % Value output
    \node at (-4.5,-69,1.5) {Evaluation: [-1, 1]};

    % Policy Head (right branch)
    \cube[]{7}{-36}{0}{\ConvColor}
    \node at (8.5,-34.5,3.5) {Policy Conv $1 \times 1 \times 32$};
    \node at (8.5,-33.5,3.5) {BatchNorm + ReLU};

    % Connect Policy Head
    \draw[->, thick] (3,-31.5,2.5) -- (8.5,-35.5,1.5);

    % Policy output
    \node at (8.5,-40,1.5) {Policy Output};
    \node at (8.5,-41.5,1.5) {(Implementation details not shown)};

    % Labels
    \node[font=\bfseries] at (-4.5,-31.5,1.5) {Value Head};
    \node[font=\bfseries] at (8.5,-31.5,1.5) {Policy Head};

    % Title
    \node[font=\bfseries, scale=1.5] at (1.5,4,2.5) {Chess Neural Network Architecture (3D View)};
\end{tikzpicture}
\end{document}
    """

    # Write the LaTeX file
    tex_file_path = os.path.join(working_dir, "chess_model_3d.tex")
    print(f"Writing LaTeX file to: {tex_file_path}")
    with open(tex_file_path, "w") as f:
        f.write(latex_content)

    # Try direct pdflatex approach, ignoring the return code
    try:
        print("Compiling with pdflatex...")
        process = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "chess_model_3d.tex"],
            cwd=working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Print compilation output for debugging
        print("pdflatex output:")
        print(process.stdout)
        if process.stderr:
            print("pdflatex errors:")
            print(process.stderr)

        # Check if PDF exists, regardless of exit code
        pdf_path = os.path.join(working_dir, "chess_model_3d.pdf")
        if os.path.exists(pdf_path):
            print(f"PDF successfully created at: {pdf_path}")
            # Rename to desired output name
            output_file = os.path.join(working_dir, "chess_model_architecture_3d.pdf")
            shutil.copy(pdf_path, output_file)
            print(f"3D neural network visualization saved to {output_file}")
            return True
        else:
            print(f"Warning: PDF not created at {pdf_path} despite compilation completing")
            return False
    except Exception as e:
        print(f"Exception during LaTeX compilation: {e}")
        return False

if __name__ == '__main__':
    print("Creating 3D visualization for ChessModel architecture...")
    success = plot_chess_neural_network_3d()
    if success:
        print("3D visualization completed successfully!")
    else:
        print("3D visualization failed!")
