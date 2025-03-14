using Plots

# Function to calculate the positions of the link ends with absolute angles
function forward_kinematics_absolute_1(theta1, theta2, l1, l2)
    x1 = l1 * cos(theta1)
    y1 = l1 * sin(theta1)
    x2 = x1 + l2 * cos(theta2)
    y2 = y1 + l2 * sin(theta2)
    return (x1, y1, x2, y2)
end

function forward_kinematics_absolute_2(theta1, theta2, l1, l2)
    x1 = l1 * cos(theta1)
    y1 = -l1 * sin(theta1)
    x2 = x1 + l2 * cos(theta2)
    y2 = y1 - l2 * sin(theta2)
    return (x1, y1, x2, y2)
end


# Function to get the corners of the rectangle representing the link
function get_rectangle_corners(x0, y0, x1, y1, width)
    dx, dy = x1 - x0, y1 - y0
    length = sqrt(dx^2 + dy^2)
    ux, uy = dx / length, dy / length
    vx, vy = -uy * width / 2, ux * width / 2
    return [(x0 + vx, y0 + vy), (x0 - vx, y0 - vy), (x1 - vx, y1 - vy), (x1 + vx, y1 + vy)]
end

# Function to get the corners of the jaw manipulator
function get_jaw_corners(x, y, theta, jaw_length, jaw_width)
    jaw1_angle = theta + π/4
    jaw2_angle = theta - π/4
    jaw1_x = x + jaw_length * cos(jaw1_angle)
    jaw1_y = y + jaw_length * sin(jaw1_angle)
    jaw2_x = x + jaw_length * cos(jaw2_angle)
    jaw2_y = y + jaw_length * sin(jaw2_angle)
    return [(x, y), (jaw1_x, jaw1_y)], [(x, y), (jaw2_x, jaw2_y)]
end

# Animation function
function animate_robots(theta1_traj, theta2_traj, theta3_traj, theta4_traj, d, l1=1.0, l2=1.0, l3= 1.0, l4 = 1.0, width=0.1, jaw_length=0.2, jaw_width=0.05)
    n_frames = length(theta1_traj)  # Number of frames in the animation
    
    # Store the tip positions for the trajectory
    tip_positions1 = [(0.0, 0.0)]  # Initial position of the first tip of robot 1
    tip_positions2 = [(0.0, 0.0)]  # Initial position of the second tip of robot 1
    tip_positions3 = [(0.0, d)]  # Initial position of the first tip of robot 2 (base shifted by d)
    tip_positions4 = [(0.0, d)]  # Initial position of the second tip of robot 2

    # Animation loop
    @gif for i in 1:n_frames
        theta1 = theta1_traj[i]
        theta2 = theta2_traj[i]
        theta3 = theta3_traj[i]
        theta4 = theta4_traj[i]
        
        # Forward kinematics for robot 1 with absolute angles
        x1, y1, x2, y2 = forward_kinematics_absolute_1(theta1, theta2, l1, l2)
        
        # Forward kinematics for robot 2 with absolute angles (base shifted by d in y direction)
        x3, y3, x4, y4 = forward_kinematics_absolute_2(theta3, theta4, l3, l4)
        y3 += d
        y4 += d
        
        link1_corners = get_rectangle_corners(0, 0, x1, y1, width)
        link2_corners = get_rectangle_corners(x1, y1, x2, y2, width)
        link3_corners = get_rectangle_corners(0, d, x3, y3, width)
        link4_corners = get_rectangle_corners(x3, y3, x4, y4, width)

        jaw1_corners1, jaw1_corners2 = get_jaw_corners(x2, y2, theta2, jaw_length, jaw_width)
        jaw2_corners1, jaw2_corners2 = get_jaw_corners(x4, y4, theta4, jaw_length, jaw_width)

        # Update the tip positions
        push!(tip_positions1, (x1, y1))
        push!(tip_positions2, (x2, y2))
        push!(tip_positions3, (x3, y3))
        push!(tip_positions4, (x4, y4))

        # Clear the plot by creating a new plot object in each iteration
        p = plot(xlim=(-2, 2), ylim=(-2, 2 + d), legend=false, aspect_ratio=:equal)
        plot!(p, [c[1] for c in link1_corners], [c[2] for c in link1_corners], seriestype=:shape, color=:blue, label="")
        plot!(p, [c[1] for c in link2_corners], [c[2] for c in link2_corners], seriestype=:shape, color=:red, label="")
        plot!(p, [c[1] for c in link3_corners], [c[2] for c in link3_corners], seriestype=:shape, color=:green, label="")
        plot!(p, [c[1] for c in link4_corners], [c[2] for c in link4_corners], seriestype=:shape, color=:orange, label="")

        # Plot the jaws
        plot!(p, [jaw1_corners1[1][1], jaw1_corners1[2][1]], [jaw1_corners1[1][2], jaw1_corners1[2][2]], color=:black, label="")
        plot!(p, [jaw1_corners2[1][1], jaw1_corners2[2][1]], [jaw1_corners2[1][2], jaw1_corners2[2][2]], color=:black, label="")
        plot!(p, [jaw2_corners1[1][1], jaw2_corners1[2][1]], [jaw2_corners1[1][2], jaw2_corners1[2][2]], color=:black, label="")
        plot!(p, [jaw2_corners2[1][1], jaw2_corners2[2][1]], [jaw2_corners2[1][2], jaw2_corners2[2][2]], color=:black, label="")

        # Plot the tips' trajectories
        scatter!(p, [pos[1] for pos in tip_positions1], [pos[2] for pos in tip_positions1], color=:black, label="Tip 1 Trajectory", markersize = 1)
        scatter!(p, [pos[1] for pos in tip_positions2], [pos[2] for pos in tip_positions2], color=:purple, label="Tip 2 Trajectory", markersize = 1)
        scatter!(p, [pos[1] for pos in tip_positions3], [pos[2] for pos in tip_positions3], color=:yellow, label="Tip 3 Trajectory", markersize = 1)
        scatter!(p, [pos[1] for pos in tip_positions4], [pos[2] for pos in tip_positions4], color=:cyan, label="Tip 4 Trajectory", markersize = 1)

        display(p)
    end every 2
end

# Example usage
theta1_traj = range(0, stop=2π, length=100)
theta2_traj = range(0, stop=π, length=100)
theta3_traj = range(0, stop=2π, length=100)
theta4_traj = range(0, stop=π, length=100)
d = 1.5  # Distance between the bases of the two robots

animate_robots(theta1_traj, theta2_traj, theta3_traj, theta4_traj, d)
