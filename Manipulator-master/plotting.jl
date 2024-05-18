using Plots

# Function to calculate the positions of the links
function forward_kinematics(theta1, theta2, l1, l2)
    x1 = l1 * cos(theta1)
    y1 = l1 * sin(theta1)
    x2 = x1 + l2 * cos(theta1 + theta2)
    y2 = y1 + l2 * sin(theta1 + theta2)
    return (x1, y1, x2, y2)
end

# Parameters
l1 = 1.0  # Length of the first link
l2 = 1.0  # Length of the second link
n_frames = 100  # Number of frames in the animation

# Define the range of joint angles for the animation
theta1_range = range(0, stop=2π, length=n_frames)
theta2_range = range(0, stop=2π, length=n_frames)

# Create a plot object
p = plot(xlim=(-2, 2), ylim=(-2, 2), legend=false)

# Animation loop
@gif for i in 1:n_frames
    theta1 = theta1_range[i]
    theta2 = theta2_range[i]
    x1, y1, x2, y2 = forward_kinematics(theta1, theta2, l1, l2)
    
    plot!(p, [0, x1], [0, y1], label="Link 1", lw=3, color=:blue)
    plot!(p, [x1, x2], [y1, y2], label="Link 2", lw=3, color=:red)
    
    scatter!([0, x1, x2], [0, y1, y2], color=:black)
end every 2

