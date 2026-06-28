# UAV Domain Vocabulary
# A list of core terms, abbreviations, and concepts related to Unmanned Aerial Vehicles (UAVs)
# used to score and filter candidate technical terms.

UAV_SEED_KEYWORDS = {
    # Core Domain Abbreviations & Names
    "uav", "uas", "uavs", "uass", "drone", "drones", "unmanned", "aerial", "vehicle", "vehicles",
    "aircraft", "airframe", "airframes", "multicopter", "multicopters", "quadrotor", "quadrotors",
    "fixed-wing", "rotorcraft", "quadcopter", "quadcopters", "hexacopter", "octocopter", "glider", "blimp",
    
    # Flight Control & Dynamics
    "autopilot", "autopilots", "guidance", "navigation", "aerodynamics", "aerodynamic",
    "attitude", "altitude", "heading", "trajectory", "trajectories", "waypoint", "waypoints",
    "flight", "stability", "stabilization", "controller", "controllers", "pid", "lqr", "kalman",
    "roll", "pitch", "yaw", "thrust", "lift", "drag", "gravity", "momentum", "torque", "inertia",
    "angle of attack", "slip", "sideslip", "vortex", "flutter",
    
    # Sensors, Avionics, & Payload
    "sensor", "sensors", "imu", "imus", "gps", "gnss", "accelerometer", "accelerometers",
    "gyroscope", "gyroscopes", "magnetometer", "magnetometers", "barometer", "barometers",
    "lidar", "radar", "sonar", "camera", "cameras", "payload", "payloads", "gimbal", "gimbals",
    "telemetry", "transceiver", "transceivers", "transmitter", "receiver", "antenna", "antennas",
    "avionics", "inertial", "measurement", "ranging", "altimeter",
    
    # Communication, Ground Systems & Operations
    "gcs", "ground control", "station", "stations", "datalink", "data link", "uplink", "downlink",
    "telecommunication", "rf", "radio", "frequency", "los", "line of sight", "bvlos", "beyond line of sight",
    "satcom", "satellite", "link", "links", "communication", "communications",
    
    # Autonomy & Algorithms
    "autonomous", "autonomy", "semi-autonomous", "obstacle", "avoidance", "collision", "path",
    "planning", "mapping", "localization", "slam", "simultaneous localization and mapping",
    "state estimation", "obstacle avoidance", "collision avoidance", "path planning",
    "trajectory planning", "waypoint navigation", "geofencing", "geofence", "failsafe",
    
    # Power, Propulsion, & Hardware
    "motor", "motors", "engine", "engines", "battery", "batteries", "propeller", "propellers",
    "rotor", "rotors", "esc", "electronic speed controller", "speed controller", "servo", "servos",
    "actuator", "actuators", "propulsion", "power distribution", "li-po", "lipo", "brushless"
}

# Domain-specific multi-word patterns that are highly characteristic of UAVs
UAV_DOMAIN_PHRASES = {
    "unmanned aerial vehicle",
    "unmanned aerial vehicles",
    "unmanned aircraft system",
    "unmanned aircraft systems",
    "ground control station",
    "ground control stations",
    "inertial measurement unit",
    "inertial measurement units",
    "fixed wing",
    "fixed-wing",
    "rotary wing",
    "rotary-wing",
    "line of sight",
    "beyond line of sight",
    "flight control",
    "flight controller",
    "flight control system",
    "obstacle avoidance",
    "collision avoidance",
    "path planning",
    "state estimation",
    "attitude estimation",
    "altitude hold",
    "vertical takeoff",
    "vertical take-off",
    "vtol",
    "takeoff and landing",
    "sensor fusion"
}

def calculate_domain_score(term: str) -> float:
    """
    Calculates a relevance score for a given term (can be single or multi-word).
    Returns a score between 0.0 and 1.0 (or higher for multiple matches).
    """
    term_lower = term.lower().strip()
    
    # Direct phrase match (highest priority)
    if term_lower in UAV_DOMAIN_PHRASES:
        return 1.0
        
    # Check words inside the term
    words = [w.strip("-,.()\"'") for w in term_lower.split()]
    words = [w for w in words if w]
    
    if not words:
        return 0.0
        
    matches = 0
    for w in words:
        # Check direct word match or prefix match for key prefixes
        if w in UAV_SEED_KEYWORDS:
            matches += 1
        elif len(w) > 3 and any(w.startswith(k) or k.startswith(w) for k in ["uav", "drone", "autopilot", "aerodyn"]):
            matches += 0.5
            
    # Calculate score as ratio of matched words to total words, weighted by matches
    if matches > 0:
        # Give higher weight to multi-word technical phrases where at least one word matches
        ratio = matches / len(words)
        return round(0.3 + 0.7 * ratio, 3)
        
    return 0.0
