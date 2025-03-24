class Config:
    # Input Columns
    TICKET_SUMMARY = 'Ticket Summary'
    INTERACTION_CONTENT = 'Interaction content'

    # Type Columns to test
    TYPE_COLS = ['y1', 'y2', 'y3', 'y4']
    CLASS_COL = 'y2'  # Default single class column
    GROUPED = 'y1'
    
    # Multi-label config
    CHAIN_LEVELS = {
        'type2': ['y2'],
        'type2_type3': ['y2', 'y3'],
        'type2_type3_type4': ['y2', 'y3', 'y4']
    }
    
    # Delimiter for combined labels
    LABEL_DELIMITER = '++'