import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

# Load the dictionary from the file
with open('../pkl/action_to_idx.pkl', 'rb') as f:
    action_to_idx = pickle.load(f)

class LSTM(nn.Module):
    def __init__(self, feature_sizes, embedding_dim=64, hidden_size=128, dropout=0.5):
        super().__init__()
        # Create an embedding for each feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=size, embedding_dim=embedding_dim)
            for size in feature_sizes
        ])
        
        # Dropout module for regularization
        self.dropout = nn.Dropout(dropout)
        
        # First LSTM layer: input is concatenated embeddings
        self.lstm1 = nn.LSTM(
            input_size=embedding_dim * len(feature_sizes),
            hidden_size=hidden_size,
            batch_first=True
        )
        # A linear projection to match the dimensions for the first residual connection
        self.residual_proj1 = nn.Linear(embedding_dim * len(feature_sizes), hidden_size)
        
        # Second LSTM layer: input and output are both hidden_size
        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        
        # Activation function
        self.activation = nn.ReLU()
        
        # Final fully-connected layer to produce logits over actions
        self.fc = nn.Linear(hidden_size, len(action_to_idx))
        
    def forward(self, x):
        # x shape: (batch_size, time_steps, num_features)
        batch_size, seq_len, num_feats = x.size()
        
        # Process each feature through its embedding
        embedded = []
        for i in range(num_feats):
            emb = self.embeddings[i](x[:, :, i])  # (batch_size, seq_len, embedding_dim)
            embedded.append(emb)
        x_emb = torch.cat(embedded, dim=-1)  # (batch_size, seq_len, embedding_dim*num_feats)
        x_emb = self.dropout(x_emb)
        
        # First LSTM layer
        out1, _ = self.lstm1(x_emb)  # (batch_size, seq_len, hidden_size)
        # Residual: project input embeddings and add to LSTM output
        res1 = self.residual_proj1(x_emb)  # (batch_size, seq_len, hidden_size)
        out1 = self.activation(out1 + res1)
        out1 = self.dropout(out1)
        
        # Second LSTM layer
        out2, _ = self.lstm2(out1)  # (batch_size, seq_len, hidden_size)
        # Residual: add the output of the first LSTM layer (out1) to the output of the second
        out2 = self.activation(out2 + out1)
        out2 = self.dropout(out2)
        
        # Use the output of the final time step for prediction
        logits = self.fc(out2[:, -1, :])  # (batch_size, num_actions)
        return logits

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('../model/model.pth', map_location=device)

    # Load mappings and parameters first
    feature_to_idx = checkpoint['feature_to_idx']
    action_to_idx = checkpoint['action_to_idx']
    idx_to_action = {v: k for k, v in action_to_idx.items()}
    features_order = checkpoint['features_order']
    feature_sizes = checkpoint['feature_sizes']
    
    # Recreate model architecture
    model = LSTM(feature_sizes=feature_sizes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, feature_to_idx, action_to_idx, idx_to_action, features_order

event_actions = [
    'session_end', 'application-window-opened', 'session_start', 'agency-dashboard::layout:render', 'agency-dashboard:::view', 'agency-dashboard::widget:render', 'agency-dashboard::configurable-table:render', '::nav-header:user-signed-out', 
    'dashboard:my-book:configurable-table:render', 'dashboard:my-book:widget:render', 'triaged-submission-list:my-book:configurable-table:render', 'triaged-submission-list:my-book::view', 'dashboard:my-book:layout:render', 'dashboard:my-book::view',
    '::nav-header:action-center-click', 'action-center:::view', 'account:::view', 'account-lines:::view', 'account-lines::layout:render', 'account-lines::widget:render', 'account-lines::configurable-table:render', ':all-accounts:configurable-table:render', 
    ':all-accounts:widget:render', ':all-accounts:layout:render', ':all-accounts::view', 'submissions:policy-definition::submit-click', 'submissions:all-policy:configurable-table:render', 'submissions:all-policy::view', 'submissions:triaged_submissions-definition::view',
    'triaged-submission:triaged_submissions-definition:layout:render', 'triaged-submission:triaged_submissions-definition::view', 'triaged-submission:triaged_submissions-definition:widget:render', 'triaged-submission-list:triaged_submissions-definition:configurable-table:render', 
    'triaged-submission-list:triaged_submissions-definition::view', 'submissions:policy-definition::view', 'submissions:policy-definition:configurable-table:render', 'submissions:policy-create::view', 'submissions:policy-create::submit-click', 'account-lines:::change-rating-click', 
    'account-property-rating:perils:configurable-table:render', 'account-property-rating:perils::view', 'action-center:::submit-click', 'action-center:action-details::view', 'action-center:::close-click', 'dashboard:my-book::action-click', 'action-center:action-details:response-form:submit-click', 
    'account-lines::templeton-docs:create-document-click', 'account-property-rating:perils:perils-table:add-click', 'account-property-rating:perils:perils-table:edit-click', 'account-property-rating:perils:perils-table:delete-click', 'dashboard:portfolio-insights:layout:render', 'dashboard:portfolio-insights::view', 
    'dashboard:portfolio-insights:widget:render', 'dashboard:my-book:recent-actions-table:action-click', 'account-auto-rating:::view', 'account-auto-rating::configurable-table:render', 'account-property-rating:perils:layers:add-click', 'account-property-rating:perils:model-request-details:save-click', 'submissions:exposures-create::submit-click', 
    'submissions:all-exposures:configurable-table:render', 'submissions:all-exposures::view', 'submissions:exposures-create::view', 'dashboard:my-book:recent-actions-table:account-click', '::configurable-table:render', '::layout:render', '::widget:render', 'EMPTY', 'submissions:all-account::view', 'submissions:all-account:configurable-table:render', 
    'submissions:account-create::view', 'account-broker-view::layout:render', 'account-broker-view:::view', 'account-broker-view::widget:render', 'agency-account::layout:render', 'agency-account:::view', 'agency-account::widget:render', 'agency-account::configurable-table:render', 
    'account-broker-view::configurable-table:render', 'submissions:all-ingest_policy_through_pd:configurable-table:render', 'submissions:all-ingest_policy_through_pd::view', 'submissions:ingest_policy_through_pd-create::view', '::nav-header:help-menu-opened', 
    'account-lines::duplicate-policy-modal:duplicate-rating', 'account-property-rating::duplicate-policy-modal:duplicate-rating', 
    'account-lines::construction-excess-rater:save-new-quote-click', 'account-lines::construction-excess-rater:create-document-click',
    '::duplicate-policy-modal:duplicate-rating', 'all-accounts:renewals:layout:render', 'all-accounts:renewals::view', 'all-accounts:renewals:configurable-table:render', 
    'all-accounts:renewals:widget:render', 'submissions:all-financial_lines::view', 'dashboard:team-insights:layout:render', 
    'dashboard:team-insights::view', 'dashboard:team-insights:widget:render', 'account-property-rating:pricing-detail:configurable-table:render',
    'account-property-rating:pricing-detail::view', 'account-property-rating:pricing-detail::open-ra-file-click',
    'account-property-rating:building-details:configurable-table:render', 'account-property-rating:building-details::view', 
    'submissions:exposures-definition::view', 'submissions:all-renewal::view', 'submissions:renewal-definition::view', 
    'all-accounts:new-business::view', 'all-accounts:new-business:layout:render', 'submissions:policy-create:configurable-table:render', 
    'submissions:renewal-create::view', 'submissions:renewal-definition::submit-click', 'submissions:renewal-create::submit-click',
    'submissions:all-renewal:configurable-table:render', 'all-accounts:new-business:accounts-table:account-click', 
    'account-lines::construction-excess-rater:modify-existing-quote-click', 'linked-email-thread-attachments:triaged_submissions-definition::document-download-click', 
    'submissions:all-auto::view', 'submissions:all-auto:configurable-table:render', 'account-workers-comp-rating:::view', 
    'account-workers-comp-rating:::change-rating-click', ':all-accounts::advanced-filters-opened', ':all-accounts:accounts-table:account-click',
    'account-broker-readonly-view::layout:render', 'account-broker-readonly-view:::view', 'account-broker-readonly-view::widget:render', 
    'triaged-submission:triaged_submissions-definition::winnability-click', 'triaged-submission:triaged_submissions-definition::appetite-click', 'assigned-email-thread:::email-thread-expansion', 
    'assigned-email-thread:::document-download-click', 'submissions:all-exposure_demo::view', 'submissions:all-exposure_demo:configurable-table:render', 'submissions:all-sashco_submission:configurable-table:render', 
    'submissions:all-sashco_submission::view', 'goals-and-rules:goals:configurable-table:render', 'goals-and-rules:goals::view', 'goals-and-rules:goal-definition::view', 
    'account-broker-readonly-view::configurable-table:render', 'submissions:all-terrorism::view', 'submissions:terrorism-create::view', 'submissions:all-terrorism:configurable-table:render', 'submissions:financial_lines-create::view', 
    'submissions:all-financial_lines:configurable-table:render', 'all-accounts:new-business:configurable-table:render', 'contacts::configurable-table:render', 'brokerage::configurable-table:render', 'brokerage::layout:render', 
    'brokerage:::view', 'brokerage::widget:render', 'complex-rules::configurable-table:render', 'classification-rules::configurable-table:render', 'rule:::view', 'rule::configurable-table:render', 
    'account-lines:::action-center-click', 'account-auto-rating:::change-rating-click', ':::account-click', 'account-property-rating::configurable-table:render', 'carriers::configurable-table:render', 
    'submissions:policy-definition::save-click', 'account-property-rating:perils:layers:delete-click', 'account-auto-rating::duplicate-policy-modal:duplicate-rating',
    'classification-rule:::view', 'classification-rule::configurable-table:render', 'submissions:policy-create::save-click', 'account-property-rating:::change-rating-click', 'goals-and-rules:rules:configurable-table:render', 
    'goals-and-rules:rules::view', 'goals-and-rules:new-rule::view', 'goals-and-rules:new-rule::close-click', 'reinsurance-binders::configurable-table:render', 'reinsurers-on-binders::configurable-table:render', 'reinsurers-on-binders:::view'
]

event_to_present_action = {
    'session_end': 'End the session',
    'application-window-opened': 'Open the application window',
    'session_start': 'Start the session',
    'agency-dashboard::layout:render': 'Render the agency dashboard layout',
    'agency-dashboard:::view': 'View the agency dashboard',
    'agency-dashboard::widget:render': 'Render the agency dashboard widget',
    'agency-dashboard::configurable-table:render': 'Render the agency dashboard configurable table',
    '::nav-header:user-signed-out': 'Sign out the user from the navigation header',
    'dashboard:my-book:configurable-table:render': 'Render the my book configurable table',
    'dashboard:my-book:widget:render': 'Render the my book widget',
    'triaged-submission-list:my-book:configurable-table:render': 'Render the triaged submission list configurable table in my book',
    'triaged-submission-list:my-book::view': 'View the triaged submission list in my book',
    'dashboard:my-book:layout:render': 'Render the my book layout',
    'dashboard:my-book::view': 'View the my book dashboard',
    '::nav-header:action-center-click': 'Click on the action center in the navigation header',
    'action-center:::view': 'View the action center',
    'account:::view': 'View the account',
    'account-lines:::view': 'View the account lines',
    'account-lines::layout:render': 'Render the account lines layout',
    'account-lines::widget:render': 'Render the account lines widget',
    'account-lines::configurable-table:render': 'Render the account lines configurable table',
    ':all-accounts:configurable-table:render': 'Render the all accounts configurable table',
    ':all-accounts:widget:render': 'Render the all accounts widget',
    ':all-accounts:layout:render': 'Render the all accounts layout',
    ':all-accounts::view': 'View the all accounts',
    'submissions:policy-definition::submit-click': 'Click to submit the policy definition',
    'submissions:all-policy:configurable-table:render': 'Render the all policy configurable table',
    'submissions:all-policy::view': 'View the all policy submissions',
    'submissions:triaged_submissions-definition::view': 'View the triaged submissions definition',
    'triaged-submission:triaged_submissions-definition:layout:render': 'Render the triaged submission layout',
    'triaged-submission:triaged_submissions-definition::view': 'View the triaged submissions definition',
    'triaged-submission:triaged_submissions-definition:widget:render': 'Render the triaged submission widget',
    'triaged-submission-list:triaged_submissions-definition:configurable-table:render': 'Render the triaged submission list configurable table',
    'triaged-submission-list:triaged_submissions-definition::view': 'View the triaged submission list',
    'submissions:policy-definition::view': 'View the policy definition',
    'submissions:policy-definition:configurable-table:render': 'Render the policy definition configurable table',
    'submissions:policy-create::view': 'View the policy create',
    'submissions:policy-create::submit-click': 'Click to submit the policy create',
    'account-lines:::change-rating-click': 'Click to change the account lines rating',
    'account-property-rating:perils:configurable-table:render': 'Render the account property rating perils configurable table',
    'account-property-rating:perils::view': 'View the account property rating perils',
    'action-center:::submit-click': 'Click to submit in the action center',
    'action-center:action-details::view': 'View the action details in the action center',
    'action-center:::close-click': 'Click to close the action center',
    'dashboard:my-book::action-click': 'Click on an action in the my book dashboard',
    'action-center:action-details:response-form:submit-click': 'Click to submit the response form in the action center',
    'account-lines::templeton-docs:create-document-click': 'Click to create a document in the account lines templeton docs',
    'account-property-rating:perils:perils-table:add-click': 'Click to add a row in the perils table',
    'account-property-rating:perils:perils-table:edit-click': 'Click to edit a row in the perils table',
    'account-property-rating:perils:perils-table:delete-click': 'Click to delete a row in the perils table',
    'dashboard:portfolio-insights:layout:render': 'Render the portfolio insights layout',
    'dashboard:portfolio-insights::view': 'View the portfolio insights',
    'dashboard:portfolio-insights:widget:render': 'Render the portfolio insights widget',
    'dashboard:my-book:recent-actions-table:action-click': 'Click on an action in the recent actions table in my book',
    'account-auto-rating:::view': 'View the account auto rating',
    'account-auto-rating::configurable-table:render': 'Render the account auto rating configurable table',
    'account-property-rating:perils:layers:add-click': 'Click to add a layer in the perils layers table',
    'account-property-rating:perils:model-request-details:save-click': 'Click to save the model request details in the perils',
    'submissions:exposures-create::submit-click': 'Click to submit the exposures create',
    'submissions:all-exposures:configurable-table:render': 'Render the all exposures configurable table',
    'submissions:all-exposures::view': 'View the all exposures submissions',
    'submissions:exposures-create::view': 'View the exposures create',
    'dashboard:my-book:recent-actions-table:account-click': 'Click on an account in the recent actions table in my book',
    '::configurable-table:render': 'Render the configurable table',
    '::layout:render': 'Render the layout',
    '::widget:render': 'Render the widget',
    'EMPTY': 'No action to perform',
    'submissions:all-account::view': 'View all accounts submissions',
    'submissions:all-account:configurable-table:render': 'Render the all account configurable table',
    'submissions:account-create::view': 'View the account create submission',
    'account-broker-view::layout:render': 'Render the account broker view layout',
    'account-broker-view:::view': 'View the account broker view',
    'account-broker-view::widget:render': 'Render the account broker view widget',
    'agency-account::layout:render': 'Render the agency account layout',
    'agency-account:::view': 'View the agency account',
    'agency-account::widget:render': 'Render the agency account widget',
    'agency-account::configurable-table:render': 'Render the agency account configurable table',
    'account-broker-view::configurable-table:render': 'Render the account broker view configurable table',
    'submissions:all-ingest_policy_through_pd:configurable-table:render': 'Render the all ingest policy through pd configurable table',
    'submissions:all-ingest_policy_through_pd::view': 'View the all ingest policy through pd submissions',
    'submissions:ingest_policy_through_pd-create::view': 'View the ingest policy through pd create submission',
    '::nav-header:help-menu-opened': 'Open the help menu in the navigation header',
    'account-lines::duplicate-policy-modal:duplicate-rating': 'Click to duplicate the rating in the duplicate policy modal',
    'account-property-rating::duplicate-policy-modal:duplicate-rating': 'Click to duplicate the rating in the account property rating duplicate policy modal',
    'account-lines::construction-excess-rater:save-new-quote-click': 'Click to save a new quote in the construction excess rater',
    'account-lines::construction-excess-rater:create-document-click': 'Click to create a document in the construction excess rater',
    '::duplicate-policy-modal:duplicate-rating': 'Click to duplicate the rating in the duplicate policy modal',
    'all-accounts:renewals:layout:render': 'Render the all accounts renewals layout',
    'all-accounts:renewals::view': 'View the all accounts renewals',
    'all-accounts:renewals:configurable-table:render': 'Render the all accounts renewals configurable table',
    'all-accounts:renewals:widget:render': 'Render the all accounts renewals widget',
    'submissions:all-financial_lines::view': 'View the all financial lines submissions',
    'dashboard:team-insights:layout:render': 'Render the team insights layout',
    'dashboard:team-insights::view': 'View the team insights',
    'dashboard:team-insights:widget:render': 'Render the team insights widget',
    'account-property-rating:pricing-detail:configurable-table:render': 'Render the pricing detail configurable table',
    'account-property-rating:pricing-detail::view': 'View the pricing detail',
    'account-property-rating:pricing-detail::open-ra-file-click': 'Click to open the RA file in the pricing detail',
    'account-property-rating:building-details:configurable-table:render': 'Render the building details configurable table',
    'account-property-rating:building-details::view': 'View the building details',
    'submissions:exposures-definition::view': 'View the exposures definition submission',
    'submissions:all-renewal::view': 'View the all renewal submission',
    'submissions:renewal-definition::view': 'View the renewal definition submission',
    'all-accounts:new-business::view': 'View the new business in all accounts',
    'all-accounts:new-business:layout:render': 'Render the new business layout in all accounts',
    'submissions:policy-create:configurable-table:render': 'Render the policy create configurable table',
    'submissions:renewal-create::view': 'View the renewal create submission',
    'submissions:renewal-definition::submit-click': 'Click to submit the renewal definition',
    'submissions:renewal-create::submit-click': 'Click to submit the renewal create',
    'submissions:all-renewal:configurable-table:render': 'Render the all renewal configurable table',
    'all-accounts:new-business:accounts-table:account-click': 'Click on an account in the new business accounts table',
    'account-lines::construction-excess-rater:modify-existing-quote-click': 'Click to modify an existing quote in the construction excess rater',
    'linked-email-thread-attachments:triaged_submissions-definition::document-download-click': 'Click to download the document from the email thread attachments',
    'submissions:all-auto::view': 'View the all auto submissions',
    'submissions:all-auto:configurable-table:render': 'Render the all auto configurable table',
    'account-workers-comp-rating:::view': 'View the account workers compensation rating',
    'account-workers-comp-rating:::change-rating-click': 'Click to change the account workers comp rating',
    ':all-accounts::advanced-filters-opened': 'Open the advanced filters in all accounts',
    ':all-accounts:accounts-table:account-click': 'Click on an account in the accounts table in all accounts',
    'account-broker-readonly-view::layout:render': 'Render the account broker readonly view layout',
    'account-broker-readonly-view:::view': 'View the account broker readonly view',
    'account-broker-readonly-view::widget:render': 'Render the account broker readonly view widget',
    'triaged-submission:triaged_submissions-definition::winnability-click': 'Click to check winnability in the triaged submission',
    'triaged-submission:triaged_submissions-definition::appetite-click': 'Click to check appetite in the triaged submission',
    'assigned-email-thread:::email-thread-expansion': 'Expand the email thread in the assigned email thread',
    'assigned-email-thread:::document-download-click': 'Click to download the document in the assigned email thread',
    'submissions:all-exposure_demo::view': 'View the exposure demo submission',
    'submissions:all-exposure_demo:configurable-table:render': 'Render the exposure demo configurable table',
    'submissions:all-sashco_submission:configurable-table:render': 'Render the sashco submission configurable table',
    'submissions:all-sashco_submission::view': 'View the sashco submission',
    'goals-and-rules:goals:configurable-table:render': 'Render the goals configurable table',
    'goals-and-rules:goals::view': 'View the goals',
    'goals-and-rules:goal-definition::view': 'View the goal definition',
    'account-broker-readonly-view::configurable-table:render': 'Render the account broker readonly view configurable table',
    'submissions:all-terrorism::view': 'View the terrorism submissions',
    'submissions:terrorism-create::view': 'View the terrorism create submission',
    'submissions:all-terrorism:configurable-table:render': 'Render the all terrorism configurable table',
    'submissions:financial_lines-create::view': 'View the financial lines create submission',
    'submissions:all-financial_lines:configurable-table:render': 'Render the all financial lines configurable table',
    'all-accounts:new-business:configurable-table:render': 'Render the new business configurable table in all accounts',
    'contacts::configurable-table:render': 'Render the contacts configurable table',
    'brokerage::configurable-table:render': 'Render the brokerage configurable table',
    'brokerage::layout:render': 'Render the brokerage layout',
    'brokerage:::view': 'View the brokerage',
    'brokerage::widget:render': 'Render the brokerage widget',
    'complex-rules::configurable-table:render': 'Render the complex rules configurable table',
    'classification-rules::configurable-table:render': 'Render the classification rules configurable table',
    'rule:::view': 'View the rule',
    'rule::configurable-table:render': 'Render the rule configurable table',
    'account-lines:::action-center-click': 'Click on the action center in the account lines',
    'account-auto-rating:::change-rating-click': 'Click to change the account auto rating',
    ':::account-click': 'Click on the account',
    'account-property-rating::configurable-table:render': 'Render the account property rating configurable table',
    'carriers::configurable-table:render': 'Render the carriers configurable table',
    'submissions:policy-definition::save-click': 'Click to save the policy definition',
    'account-property-rating:perils:layers:delete-click': 'Click to delete a layer in the perils layers',
    'account-auto-rating::duplicate-policy-modal:duplicate-rating': 'Click to duplicate the rating in the duplicate policy modal in the account auto rating',
    'classification-rule:::view': 'View the classification rule',
    'classification-rule::configurable-table:render': 'Render the classification rule configurable table',
    'submissions:policy-create::save-click': 'Click to save the policy create',
    'account-property-rating:::change-rating-click': 'Click to change the account property rating',
    'goals-and-rules:rules:configurable-table:render': 'Render the rules configurable table',
    'goals-and-rules:rules::view': 'View the rules',
    'goals-and-rules:new-rule::view': 'View the new rule',
    'goals-and-rules:new-rule::close-click': 'Click to close the new rule',
    'reinsurance-binders::configurable-table:render': 'Render the reinsurance binders configurable table',
    'reinsurers-on-binders::configurable-table:render': 'Render the reinsurers on binders configurable table',
    'reinsurers-on-binders:::view': 'View the reinsurers on binders'
}

event_to_future_action = {
    'session_end': 'End the session soon',
    'application-window-opened': 'Open the application window soon',
    'session_start': 'Start the session soon',
    'agency-dashboard::layout:render': 'Render the agency dashboard layout soon',
    'agency-dashboard:::view': 'View the agency dashboard soon',
    'agency-dashboard::widget:render': 'Render the agency dashboard widget soon',
    'agency-dashboard::configurable-table:render': 'Render the agency dashboard configurable table soon',
    '::nav-header:user-signed-out': 'Sign out the user from the navigation header soon',
    'dashboard:my-book:configurable-table:render': 'Render the my book configurable table soon',
    'dashboard:my-book:widget:render': 'Render the my book widget soon',
    'triaged-submission-list:my-book:configurable-table:render': 'Render the triaged submission list configurable table in my book soon',
    'triaged-submission-list:my-book::view': 'View the triaged submission list in my book soon',
    'dashboard:my-book:layout:render': 'Render the my book layout soon',
    'dashboard:my-book::view': 'View the my book dashboard soon',
    '::nav-header:action-center-click': 'Click on the action center in the navigation header soon',
    'action-center:::view': 'View the action center soon',
    'account:::view': 'View the account soon',
    'account-lines:::view': 'View the account lines soon',
    'account-lines::layout:render': 'Render the account lines layout soon',
    'account-lines::widget:render': 'Render the account lines widget soon',
    'account-lines::configurable-table:render': 'Render the account lines configurable table soon',
    ':all-accounts:configurable-table:render': 'Render the all accounts configurable table soon',
    ':all-accounts:widget:render': 'Render the all accounts widget soon',
    ':all-accounts:layout:render': 'Render the all accounts layout soon',
    ':all-accounts::view': 'View the all accounts soon',
    'submissions:policy-definition::submit-click': 'Click to submit the policy definition soon',
    'submissions:all-policy:configurable-table:render': 'Render the all policy configurable table soon',
    'submissions:all-policy::view': 'View the all policy submissions soon',
    'submissions:triaged_submissions-definition::view': 'View the triaged submissions definition soon',
    'triaged-submission:triaged_submissions-definition:layout:render': 'Render the triaged submission layout soon',
    'triaged-submission:triaged_submissions-definition::view': 'View the triaged submissions definition soon',
    'triaged-submission:triaged_submissions-definition:widget:render': 'Render the triaged submission widget soon',
    'triaged-submission-list:triaged_submissions-definition:configurable-table:render': 'Render the triaged submission list configurable table soon',
    'triaged-submission-list:triaged_submissions-definition::view': 'View the triaged submission list soon',
    'submissions:policy-definition::view': 'View the policy definition soon',
    'submissions:policy-definition:configurable-table:render': 'Render the policy definition configurable table soon',
    'submissions:policy-create::view': 'View the policy create soon',
    'submissions:policy-create::submit-click': 'Click to submit the policy create soon',
    'account-lines:::change-rating-click': 'Click to change the account lines rating soon',
    'account-property-rating:perils:configurable-table:render': 'Render the account property rating perils configurable table soon',
    'account-property-rating:perils::view': 'View the account property rating perils soon',
    'action-center:::submit-click': 'Click to submit in the action center soon',
    'action-center:action-details::view': 'View the action details in the action center soon',
    'action-center:::close-click': 'Click to close the action center soon',
    'dashboard:my-book::action-click': 'Click on an action in the my book dashboard soon',
    'action-center:action-details:response-form:submit-click': 'Click to submit the response form in the action center soon',
    'account-lines::templeton-docs:create-document-click': 'Click to create a document in the account lines templeton docs soon',
    'account-property-rating:perils:perils-table:add-click': 'Click to add a row in the perils table soon',
    'account-property-rating:perils:perils-table:edit-click': 'Click to edit a row in the perils table soon',
    'account-property-rating:perils:perils-table:delete-click': 'Click to delete a row in the perils table soon',
    'dashboard:portfolio-insights:layout:render': 'Render the portfolio insights layout soon',
    'dashboard:portfolio-insights::view': 'View the portfolio insights soon',
    'dashboard:portfolio-insights:widget:render': 'Render the portfolio insights widget soon',
    'dashboard:my-book:recent-actions-table:action-click': 'Click on an action in the recent actions table in my book soon',
    'account-auto-rating:::view': 'View the account auto rating soon',
    'account-auto-rating::configurable-table:render': 'Render the account auto rating configurable table soon',
    'account-property-rating:perils:layers:add-click': 'Click to add a layer in the perils layers table soon',
    'account-property-rating:perils:model-request-details:save-click': 'Click to save the model request details in the perils soon',
    'submissions:exposures-create::submit-click': 'Click to submit the exposures create soon',
    'submissions:all-exposures:configurable-table:render': 'Render the all exposures configurable table soon',
    'submissions:all-exposures::view': 'View the all exposures submissions soon',
    'submissions:exposures-create::view': 'View the exposures create soon',
    'dashboard:my-book:recent-actions-table:account-click': 'Click on an account in the recent actions table in my book soon',
    '::configurable-table:render': 'Render the configurable table soon',
    '::layout:render': 'Render the layout soon',
    '::widget:render': 'Render the widget soon',
    'EMPTY': 'No action to perform soon',
    'submissions:all-account::view': 'View all accounts submissions soon',
    'submissions:all-account:configurable-table:render': 'Render the all account configurable table soon',
    'submissions:account-create::view': 'View the account create submission soon',
    'account-broker-view::layout:render': 'Render the account broker view layout soon',
    'account-broker-view:::view': 'View the account broker view soon',
    'account-broker-view::widget:render': 'Render the account broker view widget soon',
    'agency-account::layout:render': 'Render the agency account layout soon',
    'agency-account:::view': 'View the agency account soon',
    'agency-account::widget:render': 'Render the agency account widget soon',
    'agency-account::configurable-table:render': 'Render the agency account configurable table soon',
    'account-broker-view::configurable-table:render': 'Render the account broker view configurable table soon',
    'submissions:all-ingest_policy_through_pd:configurable-table:render': 'Render the all ingest policy through pd configurable table soon',
    'submissions:all-ingest_policy_through_pd::view': 'View the all ingest policy through pd submissions soon',
    'submissions:ingest_policy_through_pd-create::view': 'View the ingest policy through pd create submission soon',
    '::nav-header:help-menu-opened': 'Open the help menu in the navigation header soon',
    'account-lines::duplicate-policy-modal:duplicate-rating': 'Click to duplicate the rating in the duplicate policy modal soon',
    'account-property-rating::duplicate-policy-modal:duplicate-rating': 'Click to duplicate the rating in the account property rating duplicate policy modal soon',
    'account-lines::construction-excess-rater:save-new-quote-click': 'Click to save a new quote in the construction excess rater soon',
    'account-lines::construction-excess-rater:create-document-click': 'Click to create a document in the construction excess rater soon',
    '::duplicate-policy-modal:duplicate-rating': 'Click to duplicate the rating in the duplicate policy modal soon',
    'all-accounts:renewals:layout:render': 'Render the all accounts renewals layout soon',
    'all-accounts:renewals::view': 'View the all accounts renewals soon',
    'all-accounts:renewals:configurable-table:render': 'Render the all accounts renewals configurable table soon',
    'all-accounts:renewals:widget:render': 'Render the all accounts renewals widget soon',
    'submissions:all-financial_lines::view': 'View the all financial lines submissions soon',
    'dashboard:team-insights:layout:render': 'Render the team insights layout soon',
    'dashboard:team-insights::view': 'View the team insights soon',
    'dashboard:team-insights:widget:render': 'Render the team insights widget soon',
    'account-property-rating:pricing-detail:configurable-table:render': 'Render the pricing detail configurable table soon',
    'account-property-rating:pricing-detail::view': 'View the pricing detail soon',
    'account-property-rating:pricing-detail::open-ra-file-click': 'Click to open the RA file in the pricing detail soon',
    'account-property-rating:building-details:configurable-table:render': 'Render the building details configurable table soon',
    'account-property-rating:building-details::view': 'View the building details soon',
    'submissions:exposures-definition::view': 'View the exposures definition submission soon',
    'submissions:all-renewal::view': 'View the all renewal submission soon',
    'submissions:renewal-definition::view': 'View the renewal definition submission soon',
    'all-accounts:new-business::view': 'View the new business in all accounts soon',
    'all-accounts:new-business:layout:render': 'Render the new business layout in all accounts soon',
    'submissions:policy-create:configurable-table:render': 'Render the policy create configurable table soon',
    'submissions:renewal-create::view': 'View the renewal create submission soon',
    'submissions:renewal-definition::submit-click': 'Click to submit the renewal definition soon',
    'submissions:renewal-create::submit-click': 'Click to submit the renewal create soon',
    'submissions:all-renewal:configurable-table:render': 'Render the all renewal configurable table soon',
    'all-accounts:new-business:accounts-table:account-click': 'Click on an account in the new business accounts table soon',
    'account-lines::construction-excess-rater:modify-existing-quote-click': 'Click to modify an existing quote in the construction excess rater soon',
    'linked-email-thread-attachments:triaged_submissions-definition::document-download-click': 'Click to download the document from the email thread attachments soon',
    'submissions:all-auto::view': 'View the all auto submissions soon',
    'submissions:all-auto:configurable-table:render': 'Render the all auto configurable table soon',
    'account-workers-comp-rating:::view': 'View the account workers compensation rating soon',
    'account-workers-comp-rating:::change-rating-click': 'Click to change the account workers comp rating soon',
    ':all-accounts::advanced-filters-opened': 'Open the advanced filters in all accounts soon',
    ':all-accounts:accounts-table:account-click': 'Click on an account in the accounts table in all accounts soon',
    'account-broker-readonly-view::layout:render': 'Render the account broker readonly view layout soon',
    'account-broker-readonly-view:::view': 'View the account broker readonly view soon',
    'account-broker-readonly-view::widget:render': 'Render the account broker readonly view widget soon',
    'triaged-submission:triaged_submissions-definition::winnability-click': 'Click to check winnability in the triaged submission soon',
    'triaged-submission:triaged_submissions-definition::appetite-click': 'Click to check appetite in the triaged submission soon',
    'assigned-email-thread:::email-thread-expansion': 'Expand the email thread in the assigned email thread soon',
    'assigned-email-thread:::document-download-click': 'Click to download the document in the assigned email thread soon',
    'submissions:all-exposure_demo::view': 'View the exposure demo submission soon',
    'submissions:all-exposure_demo:configurable-table:render': 'Render the exposure demo configurable table soon',
    'submissions:all-sashco_submission:configurable-table:render': 'Render the sashco submission configurable table soon',
    'submissions:all-sashco_submission::view': 'View the sashco submission soon',
    'goals-and-rules:goals:configurable-table:render': 'Render the goals configurable table soon',
    'goals-and-rules:goals::view': 'View the goals soon',
    'goals-and-rules:goal-definition::view': 'View the goal definition soon',
    'account-broker-readonly-view::configurable-table:render': 'Render the account broker readonly view configurable table soon',
    'submissions:all-terrorism::view': 'View the terrorism submission soon',
    'submissions:terrorism-create::view': 'View the terrorism create submission soon',
    'submissions:all-terrorism:configurable-table:render': 'Render the terrorism configurable table soon',
    'submissions:financial_lines-create::view': 'View the financial lines create submission soon',
    'submissions:all-financial_lines:configurable-table:render': 'Render the financial lines configurable table soon',
    'all-accounts:new-business:configurable-table:render': 'Render the new business configurable table in all accounts soon',
    'contacts::configurable-table:render': 'Render the contacts configurable table soon',
    'brokerage::configurable-table:render': 'Render the brokerage configurable table soon',
    'brokerage::layout:render': 'Render the brokerage layout soon',
    'brokerage:::view': 'View the brokerage soon',
    'brokerage::widget:render': 'Render the brokerage widget soon',
    'complex-rules::configurable-table:render': 'Render the complex rules configurable table soon',
    'classification-rules::configurable-table:render': 'Render the classification rules configurable table soon',
    'rule:::view': 'View the rule soon',
    'rule::configurable-table:render': 'Render the rule configurable table soon',
    'account-lines:::action-center-click': 'Click to open the action center in the account lines soon',
    'account-auto-rating:::change-rating-click': 'Click to change the account auto rating soon',
    ':::account-click': 'Click on an account soon',
    'account-property-rating::configurable-table:render': 'Render the account property rating configurable table soon',
    'carriers::configurable-table:render': 'Render the carriers configurable table soon',
    'submissions:policy-definition::save-click': 'Click to save the policy definition soon',
    'account-property-rating:perils:layers:delete-click': 'Click to delete a layer in the perils layers table soon',
    'account-auto-rating::duplicate-policy-modal:duplicate-rating': 'Click to duplicate the account auto rating in the duplicate policy modal soon',
    'classification-rule:::view': 'View the classification rule soon',
    'classification-rule::configurable-table:render': 'Render the classification rule configurable table soon',
    'submissions:policy-create::save-click': 'Click to save the policy create soon',
    'account-property-rating:::change-rating-click': 'Click to change the account property rating soon',
    'goals-and-rules:rules:configurable-table:render': 'Render the rules configurable table soon',
    'goals-and-rules:rules::view': 'View the rules soon',
    'goals-and-rules:new-rule::view': 'View the new rule soon',
    'goals-and-rules:new-rule::close-click': 'Click to close the new rule soon',
    'reinsurance-binders::configurable-table:render': 'Render the reinsurance binders configurable table soon',
    'reinsurers-on-binders::configurable-table:render': 'Render the reinsurers on binders configurable table soon',
    'reinsurers-on-binders:::view': 'View the reinsurers on binders soon'
}

DROPDOWN_FEATURES = [
    "city", "country", "device_family", 
    "language", "os_name", "user_properties_roles"
]


# Load from the checkpoint
# event_to_future_action, event_to_present_action, event_actions, DROPDOWN_FEATURES
model, feature_to_idx, action_to_idx, idx_to_action, features_order = load_model()


### - - - - Streamlit functionality - - - ###

import streamlit as st

st.markdown("""
    <style>
        @font-face {
            font-family: 'SF Mono';
            src: url('https://fonts.googleapis.com/css2?family=SF+Mono:wght@400&display=swap') format('truetype');
        }
    </style>
""", unsafe_allow_html=True)



# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_features' not in st.session_state:
    st.session_state.current_features = {feat: 0 for feat in features_order}

# Style configuration
BUTTON_STYLE = """


<style>
    div.stButton > button:first-child {
        width: 100%;
        height: 180px; /* Increased button height */
        white-space: normal;
        font-size: 1.1em; /* Larger text */
        margin: 10px;
        background: linear-gradient(125deg, rgba(173, 216, 230, 0.2), rgba(135, 206, 250, 0.2));

        # border: 2px solid #87CEFA; /* Light blue border */
        color: #00000; /* Light blue text */
        font-weight: bold; /* Bold text */
        border-radius: 5px;
        cursor: pointer;
        font-family: 'SF Mono', monospace; /* Apply SF Mono font */
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(145deg, rgba(70, 130, 180, 0.7), rgba(100, 149, 237, 0.7)); /* Darker blue gradient on hover */
        color: white; /* White text on hover */
    }
</style>

"""
st.markdown(BUTTON_STYLE, unsafe_allow_html=True)



# Sidebar for feature selection
with st.sidebar:
    # Display image from the 'images' directory
    st.image("../images/1.png", caption="", use_container_width=True)
    st.header("Feature Selection")
    current_selections = {}
    
    for feat in DROPDOWN_FEATURES:
        options = list(feature_to_idx[feat].keys())[1:]  # exclude default
        selection = st.selectbox(
            f"{feat.replace('_', ' ').title()}",
            options,
            key=feat
        )
        current_selections[feat] = feature_to_idx[feat].get(selection, 0)

import streamlit as st

# Apply custom CSS to use SF Mono font, with fallback to Google Fonts
st.markdown("""
    <style>
        /* Link to Google Fonts for Roboto */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

        body {
            font-family: 'Roboto', sans-serif;
        }
        .stTitle, .stHeader, .stSubheader {
            font-family: 'Roboto', sans-serif;
        }
    </style>
""", unsafe_allow_html=True)

# Main interface
st.title("Sequence Action Predictor")
st.markdown("""
    This model is an LSTM-based architecture designed to process sequential data and make predictions about actions based on multiple features. The model uses embeddings for each feature in the sequence, which allows it to learn a dense representation of categorical data.
""")


# Placeholder for user input
st.subheader("Input Context")
st.write("Simulate 8 actions to get a prediction for your next action!")



# Adjust the number of columns and rows based on available buttons
COLS = 4
MAX_BUTTONS = 28  # 4 columns * 14 rows
rows = [st.columns(COLS) for _ in range((len(event_actions) + COLS - 1) // COLS)]  # Dynamic rows based on number of events

button_idx = 0
for event in event_actions:
    row = button_idx // COLS
    col = button_idx % COLS
    
    
    if row < len(rows):  # Prevent index out of range for rows
        with rows[row][col]:
            # Pass unique key using event
            if st.button(event_to_present_action[event], key=f"button_{event}"):
                # Record current state + action
                record = {
                    **current_selections,
                    "event_type": action_to_idx.get(event, 0)
                }
                st.session_state.history.append(record)
                
                # Keep only last 8 steps
                if len(st.session_state.history) > 8:
                    st.session_state.history = st.session_state.history[-8:]
        button_idx += 1



### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# History visualization
st.subheader("Action Sequence Flow")
history_html = '<div style="text-align: center; margin: 20px 0; display: flex; flex-direction: column; align-items: center;">'

# Add initial CSS
st.markdown("""
<style>
.node {
    width: 160px;
    height: 170px;
    background: linear-gradient(145deg, #ffffff, #e6e6e6);
    border-radius: 85px;
    # box-shadow: 5px 5px 15px #d1d1d1;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    padding: 15px;
    margin: 20px;
}

.connector {
    width: 3px;
    height: 40px;
    background: #d1d1d1;
}
</style>
""", unsafe_allow_html=True)

# Iterate through the history to create nodes
for i, action in enumerate(st.session_state.history[-8:]):
    event = idx_to_action[action['event_type']]
    
    # Create node for each step
    history_html += f"""
    <div class="node">
        <div class="event-text" style="color: black;">{event_to_present_action[event]}</div>
    </div>
    """
    
    # Add connector if not the last item
    if i < len(st.session_state.history[-8:]) - 1:
        history_html += '<div class="connector"></div>'

history_html += '</div>'

# Render the HTML
st.markdown(history_html, unsafe_allow_html=True)

### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Prediction logic
with st.sidebar:
    if len(st.session_state.history) >= 8:
        # Prepare input tensor
        input_data = []
        for step in st.session_state.history[-8:]:
            step_features = [step.get(feat, 0) for feat in features_order]
            input_data.append(step_features)
        
        input_tensor = torch.LongTensor([input_data])
        
        # Predict
        with torch.no_grad():
            logits = model(input_tensor)
            pred_idx = torch.argmax(logits).item()
        
        predicted_event = idx_to_action.get(pred_idx, "unknown")
        future_action = event_to_future_action.get(predicted_event, "Unknown action")
        
        st.markdown(f"""
        <p style="color: white; font-size: 1.2em; margin: 10px 0; font-weight: bold;">
            Predicted Next Action
        </p>
        <p style="color: white; font-size: 1.2em; margin: 10px 0;">
            {future_action}
        </p>
        """, unsafe_allow_html=True)
    else:
        # Default message when history is less than 8
        remaining_actions = 8 - len(st.session_state.history)
        st.markdown(f"""
        <p style="color: white; font-size: 1.2em; margin: 10px 0;">
            Waiting for {remaining_actions} more actions to make a prediction...
        </p>
        """, unsafe_allow_html=True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
import streamlit as st

# Sidebar for feature selection and reset button
import streamlit as st

# Sidebar for feature selection and reset button
with st.sidebar:
    # Minimalistic button styling using HTML and CSS
    reset_button_html = """
    <style>
    .reset-button {
        width: 100%;
        height: 40px;
        background-color: transparent;
        border: 2px solid #4CAF50;
        color: #4CAF50;
        font-size: 1em;
        font-weight: bold;
        border-radius: 5px;
        text-align: center;
        display: block;
        cursor: pointer;
        transition: all 0.3s ease;
        margin: 10px 0;
    }
    .reset-button:hover {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """
    
    # Display the reset button with custom styling
    st.markdown(reset_button_html, unsafe_allow_html=True)
    
    # Create the button using HTML to handle the reset and rerun logic
    reset_button_html = """
    <button class="reset-button" onclick="window.location.reload();">Restart Sequence</button>
    """
    st.markdown(reset_button_html, unsafe_allow_html=True)

    # Reset logic handled by a button press that triggers window reload (rerun)
    if "history" in st.session_state and len(st.session_state.history) == 0:
        st.session_state.history = []
