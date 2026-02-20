"""
    plot_force_vs_distance(m::Model, d::Data; horizon=1000)

Runs the simulation for `horizon` steps and plots the Normal Force (Y)
vs Signed Distance (X) for every unique contact pair.

  - **X-Axis:** Signed Distance (Negative = Penetration).
  - **Y-Axis:** Normal Force (N).
"""
function plot_force_vs_distance(m::Model, d::Data; horizon=1000)

    # Dictionary to store (distance, force) points for each unique pair of geoms.
    # Key: Tuple of sorted geom IDs (Int32, Int32)
    # Value: Vector of Tuple(distance, force)
    contact_data = Dict{Tuple{Int32,Int32},Vector{Tuple{Float64,Float64}}}()

    println("Simulating for $horizon steps...")

    # 1. Simulation Loop
    for _ in 1:horizon
        MuJoCo.mj_step(m, d)

        # 2. Iterate over all active contacts in this time step
        for i in 1:d.ncon
            # Access the contact struct (using 1-based indexing for the array)
            c = d.contact[i]

            # Identify the pair (sort to ensure A-B is same as B-A)
            g1, g2 = minmax(c.geom1, c.geom2)
            pair_key = (g1, g2)

            # Initialize if new
            if !haskey(contact_data, pair_key)
                contact_data[pair_key] = Tuple{Float64,Float64}[]
            end

            # --- Extract Distance ---
            # c.dist is the signed distance.
            # > 0: Separated
            # < 0: Penetrating
            dist = c.dist

            # --- Extract Normal Force ---
            # The constraint force is stored in the global d.efc_force array.
            # c.efc_address is the 0-based index in the C-array where this contact starts.
            force = 0.0

            if c.efc_address >= 0
                # Convert 0-based C index to 1-based Julia index
                idx = c.efc_address + 1

                # The Normal Force is always the 1st variable of the contact constraint
                force = d.efc_force[idx]
            end

            # Only record if there is actual force or penetration relevant to the plot
            # (Optional: filter out points with 0 force and large separation if you want a cleaner plot)
            if force > 1e-6 || dist < 0.01
                push!(contact_data[pair_key], (dist, force))
            end
        end
    end

    # 3. Plotting
    p = plot(;
        title="Contact Stiffness Profile",
        xlabel="Signed Distance (m)\n(Negative = Penetration)",
        ylabel="Normal Force (N)",
        legend=:topright,
        grid=true,
        minorticks=true,
    )

    # Add a vertical line at 0 (surface boundary)
    vline!(p, [0.0]; color=:black, linestyle=:dash, label="")

    # 4. Process each pair
    for (pair, points) in contact_data
        # Sort points by distance so the line plot connects them in order (left to right)
        # This is important because simulation time doesn't necessarily mean monotonic distance.
        sort!(points; by=x -> x[1])

        # Unpack
        dists = [pt[1] for pt in points]
        forces = [pt[2] for pt in points]

        # Get readable names
        name1 = MuJoCo.mj_id2name(m, MuJoCo.mjOBJ_GEOM, pair[1])
        name2 = MuJoCo.mj_id2name(m, MuJoCo.mjOBJ_GEOM, pair[2])

        # Fallback to ID if name is missing
        n1 = isnothing(name1) ? "Geom $(pair[1])" : name1
        n2 = isnothing(name2) ? "Geom $(pair[2])" : name2

        label_str = "$n1 ↔ $n2"

        # Plot as a scatter+line to see individual solver solutions
        plot!(
            p,
            dists,
            forces;
            label=label_str,
            seriestype=:scatter,
            markersize=2,
            markerstrokewidth=0,
            alpha=0.6,
        )
    end

    if isempty(contact_data)
        annotate!(p, 0, 0, text("No contacts detected", :center))
    end

    return p
end
