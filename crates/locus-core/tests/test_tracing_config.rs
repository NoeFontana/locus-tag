use tracing::level_filters::LevelFilter;

#[test]
fn test_tracing_static_max_level() {
    // We expect release_max_level_info (as per user answer) or max_level_error (as per spec)
    // For now, let's verify it's at least restrictive enough.
    // The user said "release_max_level_info" in the questionnaire.
    
    let max_level = tracing::level_filters::STATIC_MAX_LEVEL;
    
    #[cfg(not(debug_assertions))]
    {
        // In release mode, it should be Info or more restrictive (Warn, Error, Off)
        assert!(max_level <= LevelFilter::INFO, "Static max level in release should be at most INFO, found {:?}", max_level);
    }
    
    #[cfg(debug_assertions)]
    {
        // In debug mode, it should be TRACE (default)
        assert_eq!(max_level, LevelFilter::TRACE, "Static max level in debug should be TRACE, found {:?}", max_level);
    }
}
