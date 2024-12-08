#import glob

$tomato_village_prefix = "/projects/inrg-lab/tomato/Variant-a(Multiclass Classification)/train";

%categories = ();
%cat_codes = {};
$cat_codes{"Early_blight"} = 13;
$cat_codes{"Late_blight"} = 2;
$cat_codes{"Healthy"} = 9;
$cat_codes{"Leaf Miner"} = 135;
$cat_codes{"Magnesium Deficiency"} = 136;
$cat_codes{"Nitrogen Deficiency"} = 137;
$cat_codes{"Pottassium Deficiency"} = 138;
$cat_codes{"Spotted Wilt Virus"} = 139;

@tv_train_files = glob("'$tomato_village_prefix/*/*.jpg'");
print "Got " . scalar(@tv_train_files) . " images\n";

open (my $fh, ">", "tomato.csv") or die "Can't open ./tomato.csv for writing";

print $fh "image,id\n";
foreach $image_file (@tv_train_files)
{
    if ($image_file =~ m#train/([^/]+)/\S+.jpg#)
    {
        $category = $1;
        $image_file =~ s#^.*train/##;
        $cat_code = $cat_codes{$category};
        print $fh "$image_file,$cat_code\n";
    }
    else
    {
        die "Unable to parse category for file $image_file";
    }
}

close($fh);

#tomato_village_extra_category_count = 
