using Plots
using Plots.Measures

function add_cost_trajectory!(
    plt::Plots.Plot,
    cost_trajectory::AbstractVector;
    label_name::String="Cost",
    line_color::Symbol=:blue,
    line_style::Symbol=:solid,
)
    """
    Takes an existing plot object and appends a cost trajectory curve.

    Parameters:
    - plt: The Plots.jl plot object to mutate.
    - cost_trajectory: A vector of cost values per iteration.
    - label_name: String for the legend identifying this solver run.
    - line_color: Symbol defining the color (e.g., :blue, :red).
    - line_style: Symbol defining the line style (e.g., :solid, :dot, :dot).
    """

    # Generate the X-axis data (Iteration numbers starting from 1 to N)
    iterations = 1:length(cost_trajectory)

    # Add the curve to the existing plot using the mutating plot! function
    plot!(
        plt,
        iterations,
        cost_trajectory;
        label=label_name,
        color=line_color,
        linestyle=line_style,
        linewidth=3,
        ylims=(0.3, 50),
        titlefontsize=24,
        guidefontsize=24,
        tickfontsize=18,
        legendfontsize=12,
        yaxis=:log10,
    )

    # Ensure axes are clearly labeled
    xlabel!(plt, "iLQR Iteration")
    ylabel!(plt, "Trajectory Cost")

    return plt
end

WALKER = load("walker_Jtraj.jld2")["traj"]
WALKER_CC = load("walker_Jtraj_cc.jld2")["traj"]
ANT = load("ant_Jtraj.jld2")["traj"]
ANT_CC = load("ant_Jtraj_cc.jld2")["traj"]
CHEETAH = load("halfcheetah_Jtraj.jld2")["traj"]
CHEETAH_CC = load("halfcheetah_Jtraj_cc.jld2")["traj"]

# 2. Create the base plot with a title and legend position
my_plot = plot(;
    title="Vanilla vs. CC-iLQR Convergence",
    legend=:topright,
    grid=true,
    gridalpha=0.6,
    size=(1200, 500),
    left_margin=10mm,   # Pushes the Y-axis label right, onto the screen
    bottom_margin=10mm, # Pushes the X-axis label up, onto the screen
    right_margin=5mm,
    top_margin=5mm,
)

# 3. Pass the plot and data into our function
add_cost_trajectory!(
    my_plot,
    CHEETAH;
    label_name="Vanilla Half-Cheetah",
    line_color=:dodgerblue,
    line_style=:dot,
)
add_cost_trajectory!(
    my_plot,
    CHEETAH_CC;
    label_name="CC Half-Cheetah",
    line_color=:darkblue,
    line_style=:solid,
)

add_cost_trajectory!(
    my_plot,
    WALKER;
    label_name="Vanilla Walker2d",
    line_color=:orangered2,
    line_style=:dot,
)
add_cost_trajectory!(
    my_plot,
    WALKER_CC;
    label_name="CC Walker2d",
    line_color=:orangered4,
    line_style=:solid,
)

add_cost_trajectory!(
    my_plot,
    ANT;
    label_name="Vanilla Ant",
    line_color=:mediumseagreen,
    line_style=:dot,
)
add_cost_trajectory!(
    my_plot,
    ANT_CC;
    label_name="CC Ant",
    line_color=:darkgreen,
    line_style=:solid,
)

# 4. Display the plot (in a script, you might use display(my_plot))
my_plot
