/* SOM.H */

float count_distance(int l, float stop, float *act, float *v);
void  process_som_vectors(int lrn_yn, int material);
void  dump_som_labeling(void);
void  update_som_network(int l, int x, int y, int distance);
void  clear_som_cell_counters(void);
void  dump_som_conf_matrix(void);
void  count_som_cell_winners(void);
void  clear_som_conf_matrix(void);
void  dump_soms_epoch_score(void);

float count_distance(int l, float stop, float *act, float *v)
{
   register int i;
   float hulp=0.0;
   for (i=1; i<=N[l]; i++) {
     hulp += (act[i] - v[i]) * (act[i] - v[i]);
     if (hulp>stop) { break; }
   }
   return hulp;
}

void process_som_vectors(int lrn_yn, int material)
{
  int l, x, y, win_x, win_y, update_context;
  float distance, min_distance, min_part_win_distance;
  for (l=0;l<=last_layer;l++) {
    part_win_x[l]=part_win_y[l]=winner_labeln[l]=
    win_x=win_y=use_som_error[l]=0;
    if (DIM_SOM[l]>0) {
      min_distance=min_part_win_distance=99999.0;
      for (x=1; x<=DIM_SOM[l]; x++) {
        for (y=1; y<=DIM_SOM[l]; y++) {
          if (lrn_yn==1) {
            distance=count_distance(l, min_part_win_distance,
              act[l], som_network_vector[l][x][y]);
          } else {
            distance=count_distance(l, min_distance,
              act[l], som_network_vector[l][x][y]);
          }
          /* yes no min_distance */
          if (distance<min_distance) {
            win_x=x; win_y=y;
            min_distance=distance;
            winner_labeln[l]=(int) som_cell_teller[l][x][y][0];
          }

          /* is the winner the best partial som-winner? */
          if (((lrn_yn==1) || (DUMP>0)) &&
              /* ### (SOM_ERROR_USE>DONOTHING) && */
              (som_cell_teller[l][x][y][0]==labeln) &&
              (distance<=min_part_win_distance)) {
            part_win_x[l]=x; part_win_y[l]=y;
            min_part_win_distance=distance;
          }
        }
      }

      /* book keeping and results */
      som_total_distance[l]+=(float) sqrt ((double) min_distance);
      winner_reliability[l]=som_winner_percents[l][win_x][win_y];
      if ((part_win_x[l]>0) && (part_win_y[l]>0)) {
        part_winner_reliability[l] =
          som_winner_percents[l][part_win_x[l]][part_win_y[l]];
        part_winner_dis[l]=min_part_win_distance;
        if (part_winner_reliability[l]>=RELIABILITY_THRESHOLD) {
          use_som_error[l]=1;
          if (lrn_yn==1) { som_error_use_counter[l]++; }
        }
      }

      /* update SOMs */
      if ((lrn_yn==1) && (win_x>0) &&
          (win_y>0) && (min_distance>DONOTHING)) {
        for (x=win_x-som_cont; x<=win_x+som_cont; x++) {
          for (y=win_y-som_cont; y<=win_y+som_cont; y++) {
            if (x>=1 && y>=1 && x<=DIM_SOM[l] && y<=DIM_SOM[l]) {
              if (abs(x-win_x)>abs(y-win_y)) {
                update_context=abs(x-win_x);
              } else {
                update_context=abs(y-win_y);
              }
              update_som_network(l, x, y, update_context);
            }
          }
        }
      }

      if ((material==1) && (lrn_yn==0)) {
        som_cell_teller[l][win_x][win_y][labeln]++;
      } else {
        if (winner_labeln[l]==labeln) { som_ok[l]++; }
        if (DUMP>0) som_conf_matrix[l][labeln][winner_labeln[l]]++;
      }
    }
  }
}

void dump_som_labeling (void)
{
  int x, y, l;
  long int n_examples; float weighted_reliability;
  for (l=0; l<=last_layer; l++) {
    if (DIM_SOM[l]>0) {
      weighted_reliability=0.0; n_examples=0;
      fprintf(fplog, "\n\nClass labeling SOM: %d\n\n   ", l);
      for (x=1; x<=DIM_SOM[l]; x++) {
        fprintf(fplog, "%4d", x);
      }
      for (y=1; y<=DIM_SOM[l]; y++) {
        /* label-line */
        fprintf(fplog, "\n   ");
        for (x=1; x<=DIM_SOM[l]; x++) {
          fprintf(fplog, " %3d", som_winner_percents[l][x][y]);
          weighted_reliability += 0.01 * som_winner_percents[l][x][y] *
            som_cell_teller[l][x][y][(int) som_cell_teller[l][x][y][0]];
          n_examples += som_cell_teller[l][x][y][(int) som_cell_teller[l][x][y][0]];

        }
        /* reliability-line */
        fprintf(fplog, "\n%3d", y);
        for (x=1; x<=DIM_SOM[l]; x++) {
          if (som_cell_teller[l][x][y][0]>0) {
            fprintf(fplog, " %3s", classlabel[(int) som_cell_teller[l][x][y][0]]);
          } else {
            fprintf(fplog, "    ");
          }
        }
        /* label-count-line */
        fprintf(fplog, "\n   ");
        for (x=1; x<=DIM_SOM[l]; x++) {
          fprintf(fplog, " %3ld",
            som_cell_teller[l][x][y][(int) som_cell_teller[l][x][y][0]]);
        }
      }
      if (n_examples>0) {
        fprintf(fplog, "\nweighted reliability: %6.2f\n",
                100.0*weighted_reliability/n_examples);
      } else {
        fprintf(fplog, "\nweighted reliability: n_examples=0 !!!\n");
      }
    }
  }
}

void update_som_network(int l, int x, int y, int dist)
{
  int i;
  float update_power;
  if (DUMP>1) {
    fprintf(fplog, "\nSOM%d x=%d y=%d updated (distance=%d)", l, x, y, dist);
  }
  update_power = (float) som_lr/pow((double) 2, (double) dist);
  for (i=1; i<=N[l]; i++) {
    som_network_vector[l][x][y][i] +=
      update_power * (act[l][i]-som_network_vector[l][x][y][i]);
  }
}

void clear_som_cell_counters(void)
{
  int l, x, y, c;
  for (l=0; l<=last_layer; l++) {
    for (x=1; x<=DIM_SOM[l]; x++) {
      for (y=1; y<=DIM_SOM[l]; y++) {
        som_winner_percents[l][x][y]=0;
        for (c=0; c<=n_labels; c++) {
          som_cell_teller[l][x][y][c]=0;
        }
      }
    }
  }
}

void dump_som_conf_matrix(void)
{
  int i, j, l;
  for (l=0; l<=last_layer; l++) {
    if (DIM_SOM[l]>0) {
      fprintf(fplog, "\n\nSOM-classification(%d)-confusion matrix: %5.2f\n        ",
        l, (float) (100.0*som_ok[l]/example));
      for (i=1; i<=n_labels; i++) {
        fprintf(fplog, "%s   ", classlabel[i]);
      }
      for (i=1;i<=n_labels;i++) {
        fprintf(fplog, "\n%s  ", classlabel[i]);
        for (j=1;j<=n_labels;j++) {
              fprintf(fplog, " %5ld", som_conf_matrix[l][i][j]);
        }
      }
      fprintf(fplog, "\n");
    }
  }
}

void count_som_cell_winners(void)
{
  int l, x, y, c;
  float corr_label_counter;
  float total_corr_label_counter, n_corr_wins_best_label;

  for (l=0; l<=last_layer; l++) {
    if (DIM_SOM[l]>0) {
      for (x=1; x<=DIM_SOM[l]; x++) {
        for (y=1; y<=DIM_SOM[l]; y++) {
          total_corr_label_counter=n_corr_wins_best_label=0.0;
          /* look for the maximum corrected label counter */
          for (c=1; c<=n_labels; c++) {
            /* if (fraction[c]>DONOTHING) { */
	    /* corr_label_counter=som_cell_teller[l][x][y][c]/fraction[c]; */
              corr_label_counter=som_cell_teller[l][x][y][c];
              total_corr_label_counter+=corr_label_counter;
              if (corr_label_counter>n_corr_wins_best_label) {
                som_cell_teller[l][x][y][0]=c;
                n_corr_wins_best_label=corr_label_counter;
              }
	      /* } */
          }
          if (total_corr_label_counter>0) {
            som_winner_percents[l][x][y]=
            (int) ((100.0 * n_corr_wins_best_label)/total_corr_label_counter);
          } else {
            som_winner_percents[l][x][y]=0;
          }
        }
      }
    }
  }
}

void clear_som_conf_matrix(void)
{
  int l, i, j;
  for (l=0; l<=last_layer; l++) {
    for (i=1;i<=n_labels;i++) {
      for (j=1;j<=n_labels;j++) {
        som_conf_matrix[l][i][j]=0;
      }
    }
  }
}

void dump_soms_epoch_score(void)
{
  int l;
  fprintf(fplog, "\n\nSOMs-information: ");
  for (l=0; l<=last_layer; l++) {
    if ((DIM_SOM[l]>0) && (example>0)){
      fprintf(fplog,
              "\nSOM%1d avg_eucl_dist=%6.3f cont=%d l_r=%6.3f",
              l, som_total_distance[l]/example, som_cont, som_lr);
    }
  }
}

