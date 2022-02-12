import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--raw_data_root", type=str)
    parser.add_argument("--instance_root", type=str)
    parser.add_argument("--results_root", type=str)
    parser.add_argument("--config_file", type=str)
    
    parser.add_argument("--num_walks", type=int)
    parser.add_argument("--walk_length", type=int)
    parser.add_argument("--window", type=int)
    parser.add_argument("--p", type=float)
    parser.add_argument("--q", type=float)
    
    parser.add_argument("--freeze_nodes_emb", type=int)  # bool
    parser.add_argument("--from_pretrained", type=int)  # bool
    parser.add_argument("--pretrained_model_root", type=str)
    parser.add_argument("--file_pretrained_embs", type=str)
    parser.add_argument("--operator", type=str)
    parser.add_argument("--model", type=str)
    
    parser.add_argument("--num_gcn_layer", type=int)
    parser.add_argument("--layer_sample", type=int)  # bool
    parser.add_argument("--num_layer_sample", type=str)
    parser.add_argument("--base_emb_table_device", type=str)
    parser.add_argument("--gnn_forward_device", type=str)
    parser.add_argument("--device", type=str)

    parser.add_argument("--num_combined_cluster", type=int)
    
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--topk", type=int)
    parser.add_argument("--appr_data_root", type=str)
    parser.add_argument("--friends_topk", type=int)
    parser.add_argument("--friends_appr_data_root", type=str)
    
    parser.add_argument("--graph_file_name", type=str)
    
    parser.add_argument("--partition_num", type=int)
    parser.add_argument("--partition_gen_raw_center_graph", type=int)  # bool
    
    parser.add_argument("--file_node_groups", type=str)
    
    parser.add_argument("--use_learnable_path_weights", type=int)
    parser.add_argument("--path_weights_lr", type=float)
    
    parser.add_argument("--file_base_emb_table", type=str)
    
    parser.add_argument("--src_aggr", type=str)
    parser.add_argument("--dst_aggr", type=str)
    parser.add_argument("--remove_pos_edge", type=int)
    
    parser.add_argument("--use_target_emb", type=int)
    parser.add_argument("--pred_with_one_table", type=int)
    parser.add_argument("--multi_score_mode", type=int)
    parser.add_argument("--use_influence_score", type=int)
    parser.add_argument("--use_sigmoid", type=int)
    parser.add_argument("--influence_score_lr", type=float)
    
    # seal
    parser.add_argument("--file_emb_table", type=str)
    parser.add_argument("--num_hops", type=int)
    parser.add_argument("--sample_ratio", type=float)
    parser.add_argument("--max_nodes_per_hop", type=int)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--sortpool_k", type=int)
    parser.add_argument("--lr", type=float)
    
    parser.add_argument("--self_weight", type=float)
    parser.add_argument("--topk_neighbors", type=int)
    
    parser.add_argument("--num_2_hop_neg", type=int)
    
    parser.add_argument("--use_ssl", type=int)
    parser.add_argument("--edge_dropout_ratio", type=float)
    parser.add_argument("--edge_replace_ratio", type=float)
    parser.add_argument("--tao", type=float)
    parser.add_argument("--lam", type=float)
    parser.add_argument("--lam_dict", type=str)  # e.g. "{'loss1':0.8, 'loss2':0.4, 'loss3':0.6}"
    
    parser.add_argument("--use_nodebased_dl", type=int)
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--val_test_batch_size", type=int)
    parser.add_argument("--out_emb_table_device", type=str)
    
    parser.add_argument("--ppr_data_root", type=str)
    
    parser.add_argument("--convergence_threshold", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--val_freq", type=int)
    
    parser.add_argument("--recall_val_test", type=int)
    parser.add_argument("--use_mlp", type=int)
    parser.add_argument("--each_layer_mlp", type=int)
    parser.add_argument("--final_layer_mlp", type=int)
    parser.add_argument("--scaling", type=int)
    parser.add_argument("--use_tao", type=int)
    parser.add_argument("--l2norm", type=int)
    parser.add_argument("--ffn", type=int)
    parser.add_argument("--res_ffn", type=int)
    parser.add_argument("--learnable_scalar", type=int)
    parser.add_argument("--linear", type=int)
    parser.add_argument("--cat_self", type=int)
    parser.add_argument("--degree_mlp", type=int)
    parser.add_argument("--use_bce_loss", type=int)
    
    # dfs sample
    parser.add_argument("--restart", type=float)
    parser.add_argument("--dfs_data_root", type=str)
    
    parser.add_argument("--rank_train", type=int)
    
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--use_sparse_emb", type=int)
    
    parser.add_argument("--embs_lr", type=float)
    parser.add_argument("--gnn_lr", type=float)
    parser.add_argument("--gnn_reg", type=int)
    parser.add_argument("--gnn_reg_weight", type=float)
    
    parser.add_argument("--csr_graph", type=int)
    parser.add_argument("--use_nodes_train_dl", type=int)
    parser.add_argument("--eval_on_whole_graph", type=int)
    parser.add_argument("--num_val_src", type=int)
    parser.add_argument("--num_test_src", type=int)
    
    parser.add_argument("--two_hop_pos_val_test", type=int)
    
    parser.add_argument("--test_methods", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_root", type=str)
    
    parser.add_argument("--use_scalar", type=int)
    
    parser.add_argument("--is_user_item_graph", type=int)
    parser.add_argument("--num_user", type=int)
    parser.add_argument("--use_lightgcn_residual", type=int)
    
    parser.add_argument("--file_out_emb_table", type=str)
    
    parser.add_argument("--src_pos_neg_collate", type=int)
    parser.add_argument("--num_workers", type=int)
    
    parser.add_argument("--gan_end_to_end", type=int)
    parser.add_argument("--gan", type=int)
    parser.add_argument("--gan_gen_loss_weight", type=float)
    
    parser.add_argument("--use_weighted_sampling", type=int)
    
    parser.add_argument("--str_set", type=str)
    
    (args, unknown) = parser.parse_known_args()
    parsed_results = {}
    for arg in sorted(vars(args)):
        value = getattr(args, arg)
        if value is not None:
            parsed_results[arg] = '' if value  in ['none', 'None'] else value
    return parsed_results
