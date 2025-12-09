       
def program(d, m):
	print("\n--- Welcome to the MuJoCo Simulator for the ROVI Course! ---")
	print("This window provides real-time visualization and control of your physics simulation.")
	print("Let's explore the main UI panels on the right side:\n")

	# 1) Explanation of the 'Control' Tab
	print("1) The 'Control' Tab (Top-Right):")
	print("   - By expanding the 'Control' tab, you access the actuators defined in the model.")
	print("   - For each actuator (e.g., motors for joints), you will see a slider.")
	print("   - You can click and drag these sliders to apply forces/torques in real-time.")
	print("   - This is perfect for testing how your model responds to control inputs!")

	# 2) Explanation of the 'Rendering' Tabs
	print("2) The 'Rendering' Section:")
	print("   - 'Camera': Use the dropdown to switch between fixed scene cameras.")
	print("     * 'Free': Move the camera freely with your mouse (hold right-click to rotate, scroll to zoom).")
	print("     * 'cam1': Static cameras placed in the scene (e.g., 'top', 'side', 'front').")
	print("")
	print("   - 'Label': Toggles different types of text labels overlayed on the simulation.")
	print("     * 'Body': Shows the name of each body/object.")
	print("     * 'Joint': Shows the name and type of joints.")
	print("     * 'Actuator': Shows the name of actuators.")
	print("     * 'None': Hides all labels for a clean view.")
	print("")
	print("   - 'Frame': Visualizes different coordinate frames.")
	print("     * 'None': No frames shown.")
	print("     * 'Body': Shows the local coordinate frame (x-red, y-green, z-blue) for each body.")
	print("     * 'Geom': Shows the frame for each geometric object.")
	print("")

	# 3) Explanation of the 'Group' Tabs
	print("3) The 'Group' Section:")
	print("   - This section controls the visibility of elements based on their assigned 'group'.")
	print("   - Models are often defined in groups for organization (e.g., Group 0: main model, Group 1: collision geoms, Group 2: visual geoms).")
	print("   - Toggling a group checkbox shows/hides all elements assigned to that group.")
	print("   - This is useful for:")
	print("     * Debugging: Hiding complex visual meshes to see simpler collision geometries.")
	print("     * Performance: Turning off high-detail groups to improve simulation speed.\n")


	print("Explore these options while the simulation is running! Try applying forces, and switching views.")
	print("Close the viewer window to exit this program.")





