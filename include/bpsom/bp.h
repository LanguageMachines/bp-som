/* BP.H */

float count_error(void);
float activation_function(float summ);
void  forward(void);
void  backprop(float mm);
void  dump_result(int vb_ok);
void  screen_dump_result(int vb_ok);

/* AVG and STD of the hidden units */
void reset_avg_act(void);
void update_avg_act(void);
void count_avg_act(long int teller);
void reset_std_act(void);
void update_std_act(void);
void count_std_act(long int teller);
void dump_avg_std(void);

/* prune tools */
void clear_prune_information(void);
void prune_if_possible(void);
void prune_weights(int l, int w);

float count_error(void)
{
  int i;
  float error=0.0;

  for (i=1; i<=N[last_layer]; i++) {
    error+=((target[i]-act[last_layer][i]) *
             (target[i]-act[last_layer][i]));
  }
  return error;
}

float activation_function(float summ)
{
  if (summ<-15.0) {
    return(0.0);
  } else {
    if (summ>15.0) {
      return(1.0);
    } else {
      return(1.0 / (1.0 + (float) exp(- (double) summ )));
    }
  }
}

void forward(void)
{
  int l;
  register int i, j;

  /* first an Antal hack: use the fact that most input units have
     activation 0.0 */

  act[0][0]=1.0;
  for (j=0; j<=N[1]; j++) {
    float summ=weight[0][0][j];
    for (i=0; i<PATWIDTH; i++) {
      summ+=weight[0][onact[i]][j];
    }
    act[1][j]=activation_function(summ);
  }

  for (l=1; l<last_layer; l++) {
    act[l][0]=1.0;
    for (j=0; j<=N[l+1]; j++) {
      float summ=weight[l][0][j];
      for (i=1; i<=N[l]; i++) {
        summ+=(weight[l][i][j] * act[l][i]);
      }
      act[l+1][j]=activation_function(summ);
    }
  }
}

void backprop(float mm)
{
  register int l, i;
  int j;
  float bp_error_use, bp_error, som_error=0.0;

  /* output errors */
  for (i=1; i<=N[last_layer]; i++) {
    if (fabs(target[i]-act[last_layer][i])>ERROR_LEARN_THRESHOLD)
      error[last_layer][i] =
	(act[last_layer][i]*(1.0-act[last_layer][i]) *
	 (target[i]-act[last_layer][i]));
      else
      error[last_layer][i]=0.0;
  }

  /* STEP 6 hidden errors */
  bp_error_use=1.0-SOM_ERROR_USE;
  for (l=last_layer-1; l>0; l--) {
    if (DUMP>1) {
      fprintf(fplog, "\n\nExample=%ld: ", example);
      if (use_som_error[l]==1) {
        fprintf(fplog, "som_error l=%d x=%d y=%d",
        l, part_win_x[l], part_win_y[l]);
      }
    }
    for (i=1; i<=N[l]; i++) {
      bp_error=0.0;
      for (j=1; j<=N[l+1]; j++) {
	bp_error+=(error[l+1][j] * weight[l][i][j]);
      }
      bp_error=bp_error*act[l][i]*(1.0-act[l][i]);

      /* CALCULATION  OF THE SOM-ERROR */
      if (use_som_error[l]==1) {
        som_error=0.01 * part_winner_reliability[l] *
          (som_network_vector[l][part_win_x[l]][part_win_y[l]][i] - act[l][i]);
        error[l][i]=(bp_error_use * bp_error) +
         (SOM_ERROR_USE * som_error);
      } else {
        error[l][i]=bp_error;
      }
      if (DUMP>1) {
        fprintf(fplog, "\nU%d: BP_ERROR=%8.5f SOM_ERROR=%8.5f ERROR=%8.5f",
        i, bp_error, som_error, error[l][i]);
      }
    }
  }

  /* STEP 7 en STEP 8 updating the connection weights */

  /* first an optimization hack by Antal: use the fact that most
     input units have act 0.0 */

  for (i=1; i<=N[1]; i++) {
    for (j=0; j<PATWIDTH; j++) {
      del_old[0][onact[j]][i]=(BP_LEARN_RATE * error[1][i]) +
	(mm * del_old[0][onact[j]][i]);
      weight[0][onact[j]][i]+= /* ((1.0 - WEIGHT_DECAY_RATE) * */
  	del_old[0][onact[j]][i];
    }
  }

  for (l=0; l<=last_layer-1; l++) {
    for (i=1; i<=N[l+1]; i++) {
      if (fabs(target[i]-act[last_layer][i])>ERROR_LEARN_THRESHOLD) {
      /* if (error[l+1][i]>DONOTHING) { */
	for (j=0; j<=N[l]; j++) {
	  del_old[l][j][i]=(BP_LEARN_RATE * error[l+1][i] * act[l][j]) +
	    (mm * del_old[l][j][i]);
	  weight[l][j][i]+= /* ((1.0 - WEIGHT_DECAY_RATE) * */
	    del_old[l][j][i];
	}
      }
    }
  }
}

void dump_result(int vb_ok)
{
  int l, i;
  if (DUMP_LAYER[0]>0) {
    if (vb_ok) {
      fprintf(fplog, "\n+");
    } else {
      fprintf(fplog, "\n-");
    }
    fprintf(fplog, "IN: %s ", classlabel[labeln]);
    for (l=0; l<=last_layer; l++) {
      if (DIM_SOM[l]>0) {
        fprintf(fplog, "SOM%d[%2d,%2d] R=%3d ED=%5.3f ",
          l, part_win_x[l], part_win_y[l],
          part_winner_reliability[l], sqrt((double) part_winner_dis[l]));
      }
    }
    for (i=1; i<=N[0]; i++) { fprintf(fplog, "%6.4f,", act[0][i]); }
  }

  for (l=1; l<=last_layer; l++) {
    if (DUMP_LAYER[l]>0) {
      fprintf(fplog, "\nL%d: ", l);
      for (i=1; i<=N[l]; i++) { fprintf(fplog, "%6.4f,", act[l][i]); }
    }
  }
  if (DUMP_LAYER[last_layer]>0) {
    fprintf(fplog, "\nTT: ");
    for (i=1; i<=N[last_layer]; i++) { fprintf(fplog, "%6.4f,", target[i]); }
  }
}
void screen_dump_result(int vb_ok)
{
  int l, i;
  if (vb_ok) {
    printf("\n+");
  } else {
    printf("\n-");
  }
  printf("IN: %s ", classlabel[labeln]);
  for (l=0; l<=last_layer; l++) {
    if (DIM_SOM[l]>0) {
      printf("SOM%d[%2d,%2d] R=%3d ED=%5.3f ",
        l, part_win_x[l], part_win_y[l],
        part_winner_reliability[l], sqrt((double) part_winner_dis[l]));
    }
  }
  for (i=1; i<=N[0]; i++) { printf("%6.4f,", act[0][i]); }

  for (l=1; l<=last_layer; l++) {
    printf("\nL%d: ", l);
    for (i=1; i<=N[l]; i++) { printf("%6.4f,", act[l][i]); }
  }

  printf("\nTT: ");
  for (i=1; i<=N[last_layer]; i++) { printf("%6.4f,", target[i]); }
}

void reset_avg_act(void)
{
  int l, i;
  for (l=1; l<last_layer; l++) {
    for (i=1; i<=N[l]; i++) {
      std_act[l][i]=0.0;
    }
  }
}

void update_avg_act(void)
{
  int l, i;
  for (l=1; l<last_layer; l++) {
    for (i=1; i<=N[l]; i++) {
        avg_act[l][i]+=act[l][i];
    }
  }
}

void count_avg_act(long int number)
{
  int l, i;
  for (l=1; l<last_layer; l++) {
    for (i=1; i<=N[l]; i++) {
      avg_act[l][i]=(avg_act[l][i]/number);
    }
  }
}
void reset_std_act(void)
{
  int l, i;
  for (l=1; l<last_layer; l++) {
    for (i=1; i<=N[l]; i++) { std_act[l][i]=0.0; }
  }
}

void update_std_act(void)
{
  int l, i;
  for (l=1; l<last_layer; l++) {
    for (i=1; i<=N[l]; i++) {
      std_act[l][i]+=((act[l][i]-avg_act[l][i]) *
                     (act[l][i]-avg_act[l][i]));
    }
  }
}

void count_std_act(long int teller)
{
  int l, j;
  for (l=1; l<last_layer; l++) {
    for (j=1; j<=N[l]; j++) {
      std_act[l][j]=(float) sqrt( (double) (std_act[l][j]/teller));
    }
  }
}

void dump_avg_std(void)
{
  int l, j;

  for (l=1; l<last_layer; l++) {
    fprintf(fplog, "\n\nAVG and STD layer %d:\n      ", l);
    for (j=1; j<=N[l]; j++) {
      if (std_act[l][j]<=PRUNE_THRESHOLD) {
        fprintf(fplog, "%5d*", j);
      } else {
        fprintf(fplog, "%5d ", j);
      }
    }
    fprintf(fplog, "\nAVG:  ");
    for (j=1; j<=N[l]; j++) {
      fprintf(fplog, "%5.3f ", avg_act[l][j]);
    }
    fprintf(fplog, "\nSTD:  ");
    for (j=1; j<=N[l]; j++) {
      fprintf(fplog, "%5.3f ", std_act[l][j]);
    }
    fprintf(fplog, "\n");
  }
}

void clear_prune_information(void)
{
  int l;
  pruned=0;
  for (l=1; l<last_layer; l++) { n_pruned_units[l]=0; }
}

void prune_if_possible(void)
{
  int l,i;
  for (l=1; l<last_layer; l++) {
    for (i=1; i<=N[l]; i++) {
      if ((std_act[l][i]<=PRUNE_THRESHOLD) && (N[l]>1)) {
        n_pruned_units[l]++;
        printf(" %d:-%d", l, n_pruned_units[l]);
        fprintf(fplog, " %d:-%d", l, n_pruned_units[l]);
        prune_weights(l,i);
        pruned=1;
		nothing_done=0;
      }
    }
  }
}

void prune_weights(int l, int w)
{
  int i, j, x, y;
  for (i=0; i<=N[l-1]; i++) {
    for (j=w; j<N[l]; j++) {
      weight[l-1][i][j]=weight[l-1][i][j+1];
      del_old[l-1][i][j]=del_old[l-1][i][j+1];
    }
  }
  for (i=1; i<=N[l+1]; i++) {
    weight[l][0][i]=weight[l][0][i]+
      (avg_act[l][w]*weight[l][w][i]);
  }
  for (i=w; i<N[l]; i++) {
    for (j=1; j<=N[l+1]; j++) {
      weight[l][i][j]=weight[l][i+1][j];
      del_old[l][i][j]=del_old[l][i+1][j];
    }
  }
  for (x=1; x<=DIM_SOM[l]; x++) {
    for (y=1; y<=DIM_SOM[l]; y++) {
      for (i=w; i<N[l]; i++) {
        som_network_vector[l][x][y][i]=som_network_vector[l][x][y][i+1];
      }
    }
  }
  for (i=w; i<N[l]; i++) {
    avg_act[l][i]=avg_act[l][i+1];
    std_act[l][i]=std_act[l][i+1];
  }
  N[l]=N[l]-1;
}
