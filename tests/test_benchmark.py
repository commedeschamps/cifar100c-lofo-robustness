"""Part 3 unit fixtures are synthetic software checks, never research results."""
import argparse
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn

from src.benchmark import data
from src.benchmark.checkpoints import MODELS, adapt_checkpoint
from src.model import build_model
from src.lofo import ORIGINAL_SPLIT_SHA256
from src.benchmark.download import checked_members, verify_archive
from src.benchmark.metrics import (aggregate, condition_metrics, exposure, paired_bootstrap,
                                   expected_keys, validate_complete)
from src.benchmark.runner import (cache_path, evaluation, infer, load_condition, run_lock, save_condition)
from src.config import Config
from src.dataset import make_transforms
from src.lofo import FAMILIES, REPRESENTATIVES


def synthetic_rows():
    """Counts only, ephemeral fixture; deliberately different family/type weighting."""
    rows = []
    for m in MODELS:
        for c in ("clean", *data.CORRUPTIONS):
            for s in ([0] if c == "clean" else range(1,6)):
                correct = (8000 if c == "clean" else 6000 if data.FAMILY_OF[c] == "noise" else 4000)
                if m != "clean_baseline":
                    correct += 100
                rows.append({"model": m, "seed": 42, "family": "clean" if c == "clean" else data.FAMILY_OF[c],
                             "corruption": c, "severity": s, "n_samples": 10000, "n_correct": correct,
                             "accuracy_pct": correct/100, "error_pct": 100-correct/100})
    return rows


class TinyModel(nn.Module):
    def forward(self, x):
        logits = torch.zeros(len(x),100,device=x.device)
        logits[:,0] = x.mean((1,2,3))
        return logits


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_fifteen_standard_names_and_unequal_families(self):
        self.assertEqual(tuple(map(len, FAMILIES.values())), (3,4,4,4))
        self.assertEqual(len(set(data.CORRUPTIONS)), 15)
        self.assertEqual(set(data.CORRUPTIONS), set('gaussian_noise shot_noise impulse_noise defocus_blur glass_blur motion_blur zoom_blur snow frost fog brightness contrast elastic_transform pixelate jpeg_compression'.split()))
        self.assertEqual(len(expected_keys()),380)

    def test_severity_boundaries_first_last_and_mmap(self):
        with tempfile.TemporaryDirectory(prefix="synthetic_only_") as tmp:
            pixels = np.zeros((10,32,32,3),dtype=np.uint8)
            for s in range(1,6):
                pixels[data.severity_slice(s,2)] = s
            np.save(Path(tmp)/'test.npy',pixels)
            array = np.load(Path(tmp)/'test.npy',mmap_mode='r')
            self.assertIsInstance(array,np.memmap)
            data.validate_images(array,2)
            for s in range(1,6):
                self.assertTrue(np.all(array[data.severity_slice(s,2)] == s))
            self.assertEqual(data.severity_slice(5).stop,50000)
        for s in (0,6,1.0,True):
            with self.assertRaises(ValueError):
                data.severity_slice(s)

    def test_label_alignment_all_blocks_length_type_and_range(self):
        targets = np.array([3,1,99,0])
        np.testing.assert_array_equal(data.validate_labels(np.tile(targets,5), targets),targets)
        data.validate_labels(targets.astype(np.uint8),targets)
        for wrong in (np.roll(targets,1),np.tile(targets,4),targets.astype(float),targets[:,None],np.array([3,1,100,0])):
            with self.assertRaises(ValueError):
                data.validate_labels(wrong,targets)
        damaged = np.tile(targets,5)
        damaged[7]=2
        with self.assertRaises(ValueError):
            data.validate_labels(damaged,targets)

    def test_rgb_normalization_matches_existing_transform_exactly(self):
        pixels=np.zeros((2,32,32,3),dtype=np.uint8)
        pixels[...,0],pixels[...,1],pixels[...,2]=255,127,19
        _,transform=make_transforms(Config())
        expected=torch.stack([transform(Image.fromarray(p)) for p in pixels])
        actual=data.normalize_rgb(pixels,torch.device('cpu'))
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertEqual(tuple(actual.shape),(2,3,32,32))
        for wrong in (pixels.transpose(0,3,1,2),pixels.astype(float),pixels[:,:30]):
            with self.assertRaises(ValueError):
                data.normalize_rgb(wrong,'cpu')

    def test_invalid_arrays_and_unverified_dataset_rejected(self):
        for wrong in (np.zeros((9,32,32,3),np.uint8),np.zeros((10,32,32,3),np.float32),np.zeros((10,3,32,32),np.uint8)):
            with self.assertRaises(ValueError):
                data.validate_images(wrong,2)
        with tempfile.TemporaryDirectory(prefix="synthetic_only_") as tmp:
            np.save(Path(tmp)/'labels.npy',np.zeros(2,dtype=int))
            self.assertEqual(data.discover(tmp),Path(tmp).resolve())
            with self.assertRaisesRegex(ValueError,'official archive verification'):
                data.inspect_benchmark(tmp,[0,0])
            (Path(tmp)/'official_provenance.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'provenance'):
                data.inspect_benchmark(tmp,[0,0])

    def test_accuracy_counts_partial_final_batch_and_inference_mode(self):
        images=np.full((5,32,32,3),255,dtype=np.uint8)
        labels=np.array([0,1,0,2,0])
        model=TinyModel()
        with patch.object(model,'forward',wraps=model.forward) as forward:
            pred,loss,seconds=infer(model,images,labels,torch.device('cpu'),2,False)
            self.assertEqual(forward.call_count,3)
        self.assertFalse(model.training)
        self.assertTrue(np.isfinite(loss))
        self.assertGreater(seconds,0)
        row=condition_metrics('clean_baseline','gaussian_noise',1,pred,labels,seconds,loss)
        self.assertEqual((row['n_samples'],row['n_correct'],row['accuracy_pct'],row['error_pct']),(5,3,60,40))

    def test_aggregation_weights_drops_transfer_and_direct_comparison(self):
        result=aggregate(synthetic_rows())
        baseline=result['overall'][0]
        self.assertAlmostEqual(baseline['mean_corrupted_accuracy_pct'],44)
        self.assertAlmostEqual(baseline['equal_family_accuracy_pct'],45)
        self.assertAlmostEqual(baseline['clean_drop_pp'],36)
        self.assertAlmostEqual(baseline['mean_corruption_error_pct'],56)
        self.assertEqual(len(result['corruptions']),75)
        self.assertEqual(len(result['families']),20)
        for row in result['held_out_transfer']:
            self.assertAlmostEqual(row['difference_pp'],1)
            self.assertEqual(row['model'],'lofo_'+row['family'])
        for row in result['direct_exposure']:
            self.assertAlmostEqual(row['held_out_minus_exposed_pp'],0)
            self.assertNotEqual(row['held_out_model'],row['exposed_comparator'])

    def test_incomplete_duplicate_and_bad_counts_rejected(self):
        rows=synthetic_rows()
        for wrong in (rows[:-1],rows+[rows[0]], [dict(rows[0],n_samples=32),*rows[1:]],
                      [dict(rows[0],accuracy_pct=99),*rows[1:]]):
            with self.assertRaises(ValueError):
                validate_complete(wrong)

    def test_exposure_categories_for_all_models_and_types(self):
        for family,names in FAMILIES.items():
            for c in names:
                self.assertEqual(exposure('lofo_'+family,c),'entirely_held_out_family')
                for other in FAMILIES:
                    if other!=family:
                        expected='represented_reference_operator' if c==REPRESENTATIVES[family] else 'unseen_type_within_exposed_family'
                        self.assertEqual(exposure('lofo_'+other,c),expected)
                self.assertEqual(exposure('clean_baseline',c),'clean_only_training')

    def test_bootstrap_pairing_and_no_pseudoreplication(self):
        a=np.array([[1,0,1,1],[0,1,1,1]])
        b=np.array([[0,0,1,0],[0,0,1,1]])
        one=paired_bootstrap(a,b,200,7)
        repeated=paired_bootstrap(np.tile(a,(5,1)),np.tile(b,(5,1)),200,7)
        self.assertEqual(one,repeated)  # repeated corruptions do not create extra subjects
        self.assertEqual(one['n_image_clusters'],4)
        self.assertAlmostEqual(one['difference_pp'],37.5)
        zero=paired_bootstrap(a,a,200,7)
        self.assertEqual((zero['ci95_low_pp'],zero['ci95_high_pp']),(0,0))
        with self.assertRaises(ValueError):
            paired_bootstrap(a,b[:,:3],200)

    def test_cache_atomic_roundtrip_identity_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory(prefix='synthetic_only_') as tmp:
            path=cache_path(tmp,'lofo_noise','fog',1)
            pred=np.array([1,2,3],dtype=np.uint8)
            labels=np.array([1,0,3])
            row=condition_metrics('lofo_noise','fog',1,pred,labels,1.2,.5)
            save_condition(path,'identity',row,pred)
            actual,p=load_condition(path,'identity','lofo_noise','fog',1,labels)
            self.assertEqual(actual,row)
            np.testing.assert_array_equal(p,pred)
            with self.assertRaises(ValueError):
                load_condition(path,'changed','lofo_noise','fog',1,labels)
            with self.assertRaises(ValueError):
                load_condition(path,'identity','lofo_noise','fog',1,np.zeros(3))
            path.write_bytes(b'broken')
            with self.assertRaises(ValueError):
                load_condition(path,'identity','lofo_noise','fog',1,labels)

    def test_checkpoint_adapter_rejects_unknown_schema_and_bad_normalization(self):
        with self.assertRaisesRegex(ValueError,'schema'):
            adapt_checkpoint({}, {}, 'clean_baseline', [], {})
        cfg=Config(smoke_test=False).to_dict()
        cfg['mean']=[0,0,0]
        with self.assertRaisesRegex(ValueError,'mean'):
            adapt_checkpoint({'format_version':1,'config':cfg},{'format_version':1},'clean_baseline',[],{})

    def test_strict_checkpoint_weights_output_class_order_and_completion(self):
        config=Config(smoke_test=False).to_dict()
        split={'train_indices':list(range(45000)), 'val_indices':list(range(45000,50000))}
        manifest={**split,'classes':list(range(100)),'split_sha256':ORIGINAL_SPLIT_SHA256,
                  'train_count':45000,'val_count':5000}
        history=[{'epoch':i,'val_accuracy_pct':float(i)} for i in range(1,101)]
        best={'format_version':1,'config':config,'seed':42,'epoch':100,'best_epoch':100,
              'best_validation_accuracy_pct':100.0,'data_manifest':manifest,'source_sha256':'synthetic_only',
              'model_state_dict':build_model().state_dict()}
        last={**best,'history':history}
        model=adapt_checkpoint(best,last,'clean_baseline',list(range(100)),split)
        self.assertEqual(model.fc.out_features,100)
        wrong=dict(best['model_state_dict'])
        wrong['fc.weight']=torch.zeros(99,512)
        with self.assertRaisesRegex(ValueError,'architecture'):
            adapt_checkpoint(dict(best,model_state_dict=wrong),last,'clean_baseline',list(range(100)),split)
        wrong=dict(best['model_state_dict'])
        wrong.pop('conv1.weight')
        with self.assertRaisesRegex(ValueError,'architecture'):
            adapt_checkpoint(dict(best,model_state_dict=wrong),last,'clean_baseline',list(range(100)),split)
        with self.assertRaisesRegex(ValueError,'class ordering'):
            adapt_checkpoint(best,last,'clean_baseline',list(reversed(range(100))),split)
        with self.assertRaisesRegex(ValueError,'completion'):
            adapt_checkpoint(best,dict(last,epoch=99),'clean_baseline',list(range(100)),split)

    def test_saved_metric_plots_all_five_vector_and_raster_figures(self):
        # Ephemeral synthetic rows exercise layout/export only. No scientific artifacts escape this directory.
        from src.benchmark.plots import plot_saved
        with tempfile.TemporaryDirectory(prefix='synthetic_only_plot_unit_') as tmp:
            root=Path(tmp)
            (root/'metrics').mkdir()
            rows=synthetic_rows()
            (root/'metrics/evaluation.json').write_text(json.dumps({'summary':{'benchmark_complete':True,'mode':'full',
                'fixture':'SYNTHETIC SOFTWARE TEST ONLY'},'conditions':rows,'aggregates':aggregate(rows)}))
            plot_saved(root)
            self.assertEqual(len(list((root/'plots').glob('*.svg'))),5)
            for path in (root/'plots').glob('*.png'):
                with Image.open(path) as image:
                    self.assertGreater(image.width,1500)
                    self.assertGreaterEqual(round(image.info['dpi'][0]),300)
            self.assertEqual(len(json.loads((root/'plots/index.json').read_text())['files']),10)

    def test_official_checksum_and_safe_archive_members(self):
        with tempfile.TemporaryDirectory(prefix='synthetic_only_') as tmp:
            path=Path(tmp)/'fake.tar'
            path.write_bytes(b'not the official archive')
            with self.assertRaisesRegex(ValueError,'MD5'):
                verify_archive(path,tmp)
            for bad in ('../escape','/absolute','CIFAR-100-C/../escape'):
                with tarfile.open(path,'w') as archive:
                    item=tarfile.TarInfo(bad)
                    archive.addfile(item,io.BytesIO())
                with tarfile.open(path) as archive, self.assertRaises(ValueError):
                    checked_members(archive)
            with tarfile.open(path,'w') as archive:
                item=tarfile.TarInfo('CIFAR-100-C/link')
                item.type=tarfile.SYMTYPE
                item.linkname='/tmp'
                archive.addfile(item)
            with tarfile.open(path) as archive, self.assertRaises(ValueError):
                checked_members(archive)

    def test_download_verify_and_tampered_extraction_with_tiny_synthetic_archive(self):
        from src.benchmark.download import download
        import hashlib
        # No network call or large array. Fake checksum is scoped only to this temporary software fixture.
        with tempfile.TemporaryDirectory(prefix='synthetic_only_archive_') as tmp:
            bundle=io.BytesIO()
            with tarfile.open(fileobj=bundle,mode='w') as archive:
                for name in (*data.CORRUPTIONS,'labels'):
                    encoded=io.BytesIO()
                    np.save(encoded,np.zeros((2,),dtype=np.uint8))
                    content=encoded.getvalue()
                    item=tarfile.TarInfo('CIFAR-100-C/'+name+'.npy')
                    item.size=len(content)
                    archive.addfile(item,io.BytesIO(content))
            content=bundle.getvalue()
            fake_md5=hashlib.md5(content).hexdigest()
            with patch('src.benchmark.download.ARCHIVE_MD5',fake_md5), \
                 patch('src.benchmark.download.urllib.request.urlopen',return_value=io.BytesIO(content)) as network:
                download(tmp)
                network.assert_called_once_with(data.URL,timeout=120)
                root=Path(tmp)/'CIFAR-100-C'
                receipt=json.loads((root/'official_provenance.json').read_text())
                self.assertEqual(receipt['archive_md5'],fake_md5)
                self.assertEqual(len(receipt['files']),16)
                (root/'fog.npy').write_bytes(b'tampered fixture')
                with self.assertRaisesRegex(ValueError,'differs from verified archive'):
                    verify_archive(Path(tmp)/'CIFAR-100-C.tar',root)
                with self.assertRaises(FileExistsError):
                    download(tmp)

    def test_concurrent_output_lock(self):
        with tempfile.TemporaryDirectory(prefix='synthetic_only_') as tmp, run_lock(Path(tmp)):
            with self.assertRaises(RuntimeError),run_lock(Path(tmp)):
                pass

    def test_smoke_resume_skips_completed_conditions_and_never_completes_benchmark(self):
        with tempfile.TemporaryDirectory(prefix='synthetic_only_smoke_') as tmp:
            args=argparse.Namespace(mode='smoke',output=tmp,device='cpu',batch_size=2,cpu_threads=2,
                smoke_images=4,benchmark_images=2,data_root='unused',corruption_root=None,results_root='unused',resume=False,bootstrap=0)
            clean=argparse.Namespace(data=np.full((4,32,32,3),255,np.uint8),targets=[0]*4,classes=list(range(100)))
            info={'sha256':'synthetic_only','best_clean_validation_accuracy_pct':0}
            with patch('src.benchmark.runner.load_clean',return_value=(clean,{'synthetic_only':True})), \
                 patch('src.benchmark.runner.load_run',side_effect=lambda *a:(TinyModel(),info)), \
                 patch('src.benchmark.runner.discover',side_effect=FileNotFoundError), \
                 patch('src.benchmark.runner.infer',wraps=infer) as forward:
                evaluation(args)
                self.assertEqual(forward.call_count,7) # five checks + warmup + probe
                args.resume=True
                evaluation(args)
                self.assertEqual(forward.call_count,9) # probe repeats; completed conditions do not
            summary=json.loads((Path(tmp)/'metrics/summary.json').read_text())
            self.assertFalse(summary['benchmark_complete'])
            self.assertFalse(summary['scientific_result'])
            self.assertTrue(summary['pipeline_check_only'])
            self.assertEqual(summary['completed_corruption_conditions'],0)
            from src.benchmark.plots import plot_saved
            with self.assertRaises(ValueError):
                plot_saved(tmp)

    def test_baseline_gate_stops_before_other_model_or_corruption_inference(self):
        with tempfile.TemporaryDirectory(prefix='synthetic_only_') as tmp:
            args=argparse.Namespace(mode='preflight',output=tmp,device='cpu',batch_size=100,cpu_threads=2,
                smoke_images=4,benchmark_images=2,data_root='unused',corruption_root=None,results_root='unused',resume=False,bootstrap=0)
            clean=argparse.Namespace(data=np.zeros((10000,32,32,3),np.uint8),targets=[0]*10000,classes=list(range(100)))
            info={'sha256':'synthetic_only','best_clean_validation_accuracy_pct':0}
            with patch('src.benchmark.runner.load_clean',return_value=(clean,{})), \
                 patch('src.benchmark.runner.load_run',side_effect=lambda *a:(TinyModel(),info)), \
                 patch('src.benchmark.runner.discover',side_effect=FileNotFoundError), \
                 patch('src.benchmark.runner.infer',return_value=(np.zeros(10000,np.uint8),.5,1)) as forward:
                with self.assertRaisesRegex(ValueError,'GATE FAILED'):
                    evaluation(args)
                self.assertEqual(forward.call_count,1)
            summary=json.loads((Path(tmp)/'metrics/summary.json').read_text())
            self.assertEqual(summary['status'],'failed')
            self.assertFalse(summary['baseline_gate']['passed'])
            self.assertFalse(summary['benchmark_complete'])


if __name__=='__main__':
    unittest.main()
